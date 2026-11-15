"""
BDNet: Bio-Inspired Dual-Backbone Small Object Detection Network  (PyTorch re-implementation)

Paper: Guan et al., "BDNet: Bio-Inspired Dual-Backbone Small Object Detection Network", CVPR 2026.
This is an *unofficial* implementation written from the paper text (the authors' code and the
supplementary material were not available), so ambiguous details are marked "INTERPRETATION".

Install:   pip install torch torchvision numpy pillow
Run demo:  python bdnet.py          (sanity checks + tiny training run on synthetic data)

Paper component                      -> code
-------------------------------------------------------------------------------------
Color Antagonism Module (CAM, ASE)   -> ASE, CAM
Visual Cortex Hue Enhancement (VCHM) -> VCHM
Enhanced Learnable Laplacian (ELLOM) -> ELLOM
Orientation Selective Module (OrSM)  -> esck_masks, OrSM
Feature Fusion Module (FFM, V4)      -> FFM
Color / Edge Information Pathways    -> ColorBackbone / EdgeBackbone
Full detector                        -> BDNet (dual backbone + 2 FFMs + FPN + decoupled DFL head)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import batched_nms, generalized_box_iou_loss, sigmoid_focal_loss


# ----------------------------------------------------------------------------------------
# Basic blocks
# ----------------------------------------------------------------------------------------
class ConvBNAct(nn.Sequential):
    """Conv2d -> BatchNorm -> SiLU."""

    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, g: int = 1):
        super().__init__(
            nn.Conv2d(c_in, c_out, k, s, k // 2, groups=g, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )


class Bottleneck(nn.Module):
    """Two 3x3 convs with a residual connection (adds a little capacity per stage)."""

    def __init__(self, c: int):
        super().__init__()
        self.cv1 = ConvBNAct(c, c, 3)
        self.cv2 = ConvBNAct(c, c, 3)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


# ----------------------------------------------------------------------------------------
# Color Information Pathway: CAM (LGN/V1) and VCHM (V2)
# ----------------------------------------------------------------------------------------
class ASE(nn.Module):
    """
    Adaptive Selection Enhancement.

    R,G,B play the role of L,M,S cones. There are 6 ordered (excitation+, inhibition-) pairs:
    L+M-, L+S-, M+L-, M+S-, S+L-, S+M-.  Three learnable scalars [alpha, beta, gamma] are sorted
    into (max, median, min) = (delta+, delta, delta-). For each pair the excitation channel gets
    +delta+, the inhibition channel gets -delta-, the remaining channel gets +delta
    (INTERPRETATION of Fig. 2-A2: the 3! permutations of the triple over the three channels).
    Returns a (6, 3) per-pair channel-weight matrix.
    """

    PAIRS = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]  # (excitation, inhibition)

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([1.0, 0.7, 0.4]))
        exc, inh, oth = torch.zeros(6, 3), torch.zeros(6, 3), torch.zeros(6, 3)
        for p, (e, i) in enumerate(self.PAIRS):
            exc[p, e], inh[p, i], oth[p, 3 - e - i] = 1, 1, 1
        self.register_buffer("exc", exc)
        self.register_buffer("inh", inh)
        self.register_buffer("oth", oth)

    def forward(self) -> torch.Tensor:
        d_max, d_med, d_min = torch.sort(self.theta, descending=True).values
        return d_max * self.exc - d_min * self.inh + d_med * self.oth


class CAM(nn.Module):
    """
    Color Antagonism Module. Pair features of the same excitation channel are summed, then
    fused by a (grouped) convolution -> one feature group per excitation channel (R, G, B).
    Summing two weighted copies of x is linear, so it is folded into one weight per channel.
    """

    def __init__(self, out_channels: int = 3):
        super().__init__()
        assert out_channels % 3 == 0
        self.ase = ASE()
        self.fuse = nn.Sequential(
            nn.Conv2d(9, out_channels, 3, padding=1, groups=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.residual = out_channels == 3

    def forward(self, x):  # x: (B,3,H,W) RGB
        w = self.ase().view(3, 2, 3).sum(dim=1)            # (3 groups, 3 channels)
        g = x.unsqueeze(1) * w.view(1, 3, 3, 1, 1)         # (B,3,3,H,W)
        y = self.fuse(g.flatten(1, 2))                     # (B,out,H,W)
        return x + y if self.residual else y


class VCHM(nn.Module):
    """
    Visual Cortex Hue Enhancement Module: grouped conv -> pairwise channel coupling -> embedding.
      1. X' = depthwise conv(X)                       ("brightness adjustment", C groups)
      2. Y  = Conv1x1(X'),  kernel = Jordan-form mask [1,1,0,..] (shifted per row) * learnable W
              -> Y_i = w0*x'_i + w1*x'_{i+1}, giving C-1 new "hue" channels      (Eq. 1)
      3. E(X,Y) interleaves X (even slots) and Y (odd slots)  -> 2C-1 channels   (Eq. 2)
    """

    def __init__(self, c: int):
        super().__init__()
        assert c >= 2
        self.c, self.out_channels = c, 2 * c - 1
        self.dw = ConvBNAct(c, c, 3, g=c)
        self.weight = nn.Parameter(torch.empty(c - 1, c, 1, 1).uniform_(0.25, 0.75))
        mask = torch.zeros(c - 1, c, 1, 1)
        for i in range(c - 1):
            mask[i, i] = mask[i, i + 1] = 1.0
        self.register_buffer("mask", mask)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        xp = self.dw(x)
        hue = F.conv2d(xp, self.weight * self.mask)                      # (B,C-1,H,W)
        hue = torch.cat([hue, hue.new_zeros(hue.shape[0], 1, *hue.shape[2:])], 1)  # pad to C
        out = torch.stack([x, hue], dim=2).flatten(1, 2)[:, :-1]         # x0,y0,x1,y1,...,x_{C-1}
        return self.act(self.bn(out))


# ----------------------------------------------------------------------------------------
# Edge Information Pathway: ELLOM and OrSM (ESCK)
# ----------------------------------------------------------------------------------------
class ELLOM(nn.Module):
    """
    Enhanced Learnable Laplacian Operator Module (Eq. 3, the two "latter" kernels):
        [[w1,0,w3],[0,-S,0],[w7,0,w9]]   and   [[0,w2,0],[w4,-S,w6],[0,w8,0]]
    where S is the sum of the surrounding weights (zero-sum kernels, like a Laplacian).
    Initialised at w=1, i.e. the classic Laplacian kernels. Applied per channel.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.c = in_channels
        self.w_diag = nn.Parameter(torch.ones(in_channels, 4))   # w1, w3, w7, w9
        self.w_cross = nn.Parameter(torch.ones(in_channels, 4))  # w2, w4, w6, w8
        place_d, place_c, center = torch.zeros(4, 9), torch.zeros(4, 9), torch.zeros(9)
        for j, (d, c) in enumerate(zip([0, 2, 6, 8], [1, 3, 5, 7])):
            place_d[j, d], place_c[j, c] = 1, 1
        center[4] = 1
        self.register_buffer("place_d", place_d)
        self.register_buffer("place_c", place_c)
        self.register_buffer("center", center)
        self.proj = ConvBNAct(2 * in_channels, out_channels, 1)

    def _kernel(self, w, place):
        k = w @ place - w.sum(1, keepdim=True) * self.center       # (C,9), centre = -sum
        return k.view(self.c, 1, 3, 3)

    def forward(self, x):
        d = F.conv2d(x, self._kernel(self.w_diag, self.place_d), padding=1, groups=self.c)
        c = F.conv2d(x, self._kernel(self.w_cross, self.place_c), padding=1, groups=self.c)
        return self.proj(torch.cat([d.abs(), c.abs()], 1))          # edge map (B,out,H,W)


# Neighbour offsets (dy, dx), clockwise starting from "up".
_RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def esck_masks():
    """
    Eight Enhancement-Suppression kernel masks, shape (8, 9) each.
    INTERPRETATION of Fig. 3(B): compass-style kernels. For direction k the 'enhancement' side
    is the 3 ring pixels facing direction k, the 'suppression' side is the opposite 3 pixels.
    """
    pos, neg = torch.zeros(8, 9), torch.zeros(8, 9)
    for k in range(8):
        for d in (-1, 0, 1):
            dy, dx = _RING[(k + d) % 8]
            pos[k, (dy + 1) * 3 + dx + 1] = 1
            dy, dx = _RING[(k + 4 + d) % 8]
            neg[k, (dy + 1) * 3 + dx + 1] = 1
    return pos, neg


class OrSM(nn.Module):
    """
    Orientation Selective Module (V1 orientation-tuned neurons).
      * ESCK: 8 learnable directional zero-sum kernels per channel (weight-based differences:
              enhancement-side weights normalised to +1, suppression-side to -1).
      * A sub-network (1x1 conv -> 3x3 conv -> sigmoid*7 -> round, "S7R") predicts a direction
        index 0..7 per pixel -> 8 binary masks -> only the most salient direction survives, so
        kernels never superimpose.  Rounding is non-differentiable, so a straight-through
        estimator (soft Gaussian-like relaxation in the backward pass) is used (my addition).
    """

    def __init__(self, c: int, tau: float = 0.5):
        super().__init__()
        self.c, self.tau = c, tau
        pos, neg = esck_masks()
        self.register_buffer("pos", pos)
        self.register_buffer("neg", neg)
        self.raw = nn.Parameter(torch.zeros(c, 8, 9))               # exp(raw)=1 at init
        self.index_net = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Conv2d(1, 1, 3, padding=1))
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def kernels(self):
        w = self.raw.exp()
        p, n = w * self.pos, w * self.neg
        k = p / p.sum(-1, keepdim=True) - n / n.sum(-1, keepdim=True)   # zero-sum, (C,8,9)
        return k.reshape(self.c * 8, 1, 3, 3)

    def direction_mask(self, x):
        s = torch.sigmoid(self.index_net(x)) * 7                        # (B,1,H,W) in [0,7]
        hard = F.one_hot(s.round().long().squeeze(1), 8).permute(0, 3, 1, 2).to(x.dtype)
        levels = torch.arange(8, device=x.device, dtype=x.dtype).view(1, 8, 1, 1)
        soft = torch.softmax(-((s - levels) ** 2) / self.tau, dim=1)
        return hard + soft - soft.detach()                              # one-hot fwd, soft bwd

    def forward(self, x):
        B, C, H, W = x.shape
        resp = F.conv2d(x, self.kernels(), padding=1, groups=C).view(B, C, 8, H, W)
        out = (resp * self.direction_mask(x).unsqueeze(1)).sum(2)       # select 1 of 8 per pixel
        return x + self.act(self.bn(out))


# ----------------------------------------------------------------------------------------
# Feature Fusion Module (V4)
# ----------------------------------------------------------------------------------------
class FFM(nn.Module):
    """
    Injects colour information Z1 into edge features Z2.
      p = score(GAP(Z1)),  q = score(GAP(Z2))                    (per-channel importance)
      R = p q^T  (C x C, rank-1)  used as a 1x1 conv kernel on Z2   (Eq. 4-6)
      Output = Z1 || (Output_1 || ... || Output_C + Z2)             (Eq. 7)  -> 2C channels
    Because R is rank-1, Output_k = p_k * sum_i q_i Q_i, so the fast path never builds R.
    Stabilisation choices (not in the paper): p uses sigmoid, q uses softmax.
    """

    def __init__(self, c: int, out_channels: int | None = None):
        super().__init__()
        self.score1 = nn.Conv2d(c, c, 1)
        self.score2 = nn.Conv2d(c, c, 1)
        self.proj = ConvBNAct(2 * c, out_channels or 2 * c, 1)

    def injected(self, z1, z2, explicit: bool = False):
        p = torch.sigmoid(self.score1(F.adaptive_avg_pool2d(z1, 1)).flatten(1))   # (B,C)
        q = torch.softmax(self.score2(F.adaptive_avg_pool2d(z2, 1)).flatten(1), 1)  # (B,C)
        if explicit:  # literal Eq. (4)-(6): build R and apply it as a 1x1 conv kernel
            R = p.unsqueeze(2) * q.unsqueeze(1)                                      # (B,C,C)
            return torch.einsum("bki,bihw->bkhw", R, z2)
        s = torch.einsum("bi,bihw->bhw", q, z2)
        return p[:, :, None, None] * s.unsqueeze(1)

    def forward(self, z1, z2):
        return self.proj(torch.cat([z1, self.injected(z1, z2) + z2], 1))


# ----------------------------------------------------------------------------------------
# Dual backbone, neck, head
# ----------------------------------------------------------------------------------------
class ColorBackbone(nn.Module):
    """RGB -> CAM -> P1 -> P2 -> VCHM -> P3 -> VCHM -> P4   (strides 2, 4, 8, 16)."""

    def __init__(self, chs=(16, 32, 64, 128)):
        super().__init__()
        c1, c2, c3, c4 = chs
        self.cam = CAM(3)
        self.p1 = ConvBNAct(3, c1, 3, 2)
        self.p2 = nn.Sequential(ConvBNAct(c1, c2, 3, 2), Bottleneck(c2))
        self.vchm3 = VCHM(c2)
        self.p3 = nn.Sequential(ConvBNAct(self.vchm3.out_channels, c3, 3, 2), Bottleneck(c3))
        self.vchm4 = VCHM(c3)
        self.p4 = nn.Sequential(ConvBNAct(self.vchm4.out_channels, c4, 3, 2), Bottleneck(c4))

    def forward(self, x):
        p1 = self.p1(self.cam(x))
        p2 = self.p2(p1)
        p3 = self.p3(self.vchm3(p2))
        p4 = self.p4(self.vchm4(p3))
        return p2, p3, p4


class EdgeBackbone(nn.Module):
    """RGB -> ELLOM -> P1 -> P2 -> OrSM -> P3 -> OrSM -> P4  (strides 2, 4, 8, 16)."""

    def __init__(self, chs=(16, 32, 64, 128)):
        super().__init__()
        c1, c2, c3, c4 = chs
        self.ellom = ELLOM(3, 3)
        self.p1 = ConvBNAct(3, c1, 3, 2)
        self.p2 = nn.Sequential(ConvBNAct(c1, c2, 3, 2), Bottleneck(c2))
        self.orsm3 = OrSM(c2)
        self.p3 = nn.Sequential(ConvBNAct(c2, c3, 3, 2), Bottleneck(c3))
        self.orsm4 = OrSM(c3)
        self.p4 = nn.Sequential(ConvBNAct(c3, c4, 3, 2), Bottleneck(c4))

    def forward(self, x):
        p1 = self.p1(self.ellom(x))
        p2 = self.p2(p1)
        p3 = self.p3(self.orsm3(p2))
        p4 = self.p4(self.orsm4(p3))
        return p2, p3, p4


class FPN(nn.Module):
    """Top-down feature pyramid (features ordered high-res -> low-res)."""

    def __init__(self, in_chs, c: int = 64):
        super().__init__()
        self.lat = nn.ModuleList(ConvBNAct(ci, c, 1) for ci in in_chs)
        self.smooth = nn.ModuleList(ConvBNAct(c, c, 3) for _ in in_chs)

    def forward(self, feats):
        lat = [l(f) for l, f in zip(self.lat, feats)]
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[-2:], mode="nearest")
        return [s(x) for s, x in zip(self.smooth, lat)]


def make_anchors(feats, strides):
    """Anchor points (grid units, cell centres) and per-point strides for all levels."""
    pts, st = [], []
    for f, s in zip(feats, strides):
        h, w = f.shape[-2:]
        ys = torch.arange(h, device=f.device, dtype=f.dtype) + 0.5
        xs = torch.arange(w, device=f.device, dtype=f.dtype) + 0.5
        sy, sx = torch.meshgrid(ys, xs, indexing="ij")
        pts.append(torch.stack((sx, sy), -1).view(-1, 2))
        st.append(torch.full((h * w, 1), float(s), device=f.device, dtype=f.dtype))
    return torch.cat(pts), torch.cat(st)


class DetectHead(nn.Module):
    """Anchor-free decoupled head with Distribution Focal Loss style box regression
    (the paper's output is `4*reg_max` box channels + `nc` class channels)."""

    def __init__(self, nc: int, ch: int = 64, strides=(4, 8, 16), reg_max: int = 16):
        super().__init__()
        self.nc, self.reg_max, self.strides = nc, reg_max, strides
        c_box, c_cls = max(16, ch // 4, 4 * reg_max), max(ch, min(nc, 100))
        self.box_branches = nn.ModuleList(
            nn.Sequential(ConvBNAct(ch, c_box, 3), ConvBNAct(c_box, c_box, 3), nn.Conv2d(c_box, 4 * reg_max, 1))
            for _ in strides)
        self.cls_branches = nn.ModuleList(
            nn.Sequential(ConvBNAct(ch, c_cls, 3), ConvBNAct(c_cls, c_cls, 3), nn.Conv2d(c_cls, nc, 1))
            for _ in strides)
        self.register_buffer("proj", torch.arange(reg_max, dtype=torch.float32), persistent=False)
        for b, c, s in zip(self.box_branches, self.cls_branches, strides):  # YOLO-style bias init
            b[-1].bias.data[:] = 1.0
            c[-1].bias.data[:] = math.log(5 / nc / (640 / s) ** 2)

    def forward(self, feats):
        return [torch.cat([b(f), c(f)], 1) for f, b, c in zip(feats, self.box_branches, self.cls_branches)]

    def decode(self, raw):
        """raw list -> (boxes xyxy in input pixels (B,N,4), class probabilities (B,N,nc))."""
        anchors, stride_t = make_anchors(raw, self.strides)
        flat = torch.cat([r.flatten(2) for r in raw], 2)
        box, cls = flat.split([4 * self.reg_max, self.nc], 1)
        B, _, N = box.shape
        prob = box.reshape(B, 4, self.reg_max, N).softmax(2)
        ltrb = torch.einsum("bkrn,r->bnk", prob, self.proj)               # (B,N,4) grid units
        a = anchors.unsqueeze(0)
        boxes = torch.cat([a - ltrb[..., :2], a + ltrb[..., 2:]], -1) * stride_t.unsqueeze(0)
        return boxes, cls.permute(0, 2, 1).sigmoid()


class BDNet(nn.Module):
    """
    Bio-inspired dual-backbone detector.

      RGB --> ColorBackbone (CAM, VCHM) --> C2, C3, C4 --+
      RGB --> EdgeBackbone  (ELLOM, OrSM) -> E2, E3, E4 --+
      FFM #1 : (C2, E2)                                      -> F2   (stride 4)
      FFM #2 : (cat(C3, up C4), cat(E3, up E4))              -> F3   (stride 8)
      F4     : 1x1 conv of cat(C4, E4)                       -> F4   (stride 16)
      FPN -> decoupled head at strides 4 / 8 / 16.
    (Only two FFMs are used, as in Fig. 1. The F4 branch and 3-level head are my interpretation
     of "a feature pyramid network aggregates multi-scale features".)
    Input height/width must be divisible by 16.
    """

    def __init__(self, nc: int = 10, chs=(16, 32, 64, 128), neck_ch: int = 64, reg_max: int = 16):
        super().__init__()
        c1, c2, c3, c4 = chs
        self.color, self.edge = ColorBackbone(chs), EdgeBackbone(chs)
        self.ffm_p2 = FFM(c2, out_channels=2 * c2)
        self.ffm_p34 = FFM(c3 + c4, out_channels=2 * c3)
        self.p4_fuse = ConvBNAct(2 * c4, c4, 1)
        self.neck = FPN((2 * c2, 2 * c3, c4), neck_ch)
        self.head = DetectHead(nc, neck_ch, (4, 8, 16), reg_max)

    def forward(self, x):
        c2, c3, c4 = self.color(x)
        e2, e3, e4 = self.edge(x)
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="nearest")
        f2 = self.ffm_p2(c2, e2)
        f3 = self.ffm_p34(torch.cat([c3, up(c4, c3)], 1), torch.cat([e3, up(e4, e3)], 1))
        f4 = self.p4_fuse(torch.cat([c4, e4], 1))
        return self.head(self.neck([f2, f3, f4]))          # list of (B, 4*reg_max+nc, H, W)

    @torch.no_grad()
    def predict(self, x, conf_thres=0.25, iou_thres=0.6, max_det=300):
        """Call model.eval() first. Returns a list of (n,6) tensors: x1,y1,x2,y2,score,class."""
        boxes, scores = self.head.decode(self(x))
        out = []
        for b in range(x.shape[0]):
            sc, cls = scores[b].max(1)
            keep = sc > conf_thres
            bx, sc, cls = boxes[b][keep], sc[keep], cls[keep]
            idx = batched_nms(bx, sc, cls, iou_thres)[:max_det]
            out.append(torch.cat([bx[idx], sc[idx, None], cls[idx, None].float()], 1))
        return out


# ----------------------------------------------------------------------------------------
# Simplified training loss (FCOS-style assignment + GIoU + DFL + focal)
# ----------------------------------------------------------------------------------------
class SimpleDetectionLoss(nn.Module):
    """
    A compact, self-contained loss so the model can be trained without extra frameworks.
    It is NOT the TAL/CIoU loss used by YOLO12/Ultralytics - for paper-level numbers plug the
    BDNet features into an Ultralytics `Detect` head and `v8DetectionLoss` instead.

    targets: list (one per image) of (M,5) tensors [class, x1, y1, x2, y2] in input pixels.
    Each GT is assigned to ONE pyramid level by its max side (scale_ranges), then to anchor
    points that lie inside the box and within `radius` strides of its centre (nearest point
    as fallback, so tiny objects are never dropped).
    """

    def __init__(self, nc, strides=(4, 8, 16), reg_max=16, scale_ranges=((0, 32), (32, 96), (96, 1e8)),
                 radius=2.5, w_box=5.0, w_cls=1.0, w_dfl=0.5):
        super().__init__()
        self.nc, self.strides, self.reg_max, self.ranges = nc, strides, reg_max, scale_ranges
        self.radius, self.w_box, self.w_cls, self.w_dfl = radius, w_box, w_cls, w_dfl
        self.register_buffer("stride_vals", torch.tensor(strides, dtype=torch.float32), persistent=False)
        self.register_buffer("proj", torch.arange(reg_max, dtype=torch.float32), persistent=False)

    def _assign(self, pts, stride, gt):
        N, M = pts.shape[0], gt.shape[0]
        cx, cy = (gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2
        wh = gt[:, 2:] - gt[:, :2]
        size = wh.max(1).values
        lvl = torch.zeros(M, dtype=torch.long, device=gt.device)
        for i, (lo, hi) in enumerate(self.ranges):
            lvl[(size >= lo) & (size < hi)] = i
        gt_stride = self.stride_vals[lvl]                                   # (M,)
        px, py = pts[:, :1], pts[:, 1:]                                     # (N,1)
        inside = (px > gt[:, 0]) & (px < gt[:, 2]) & (py > gt[:, 1]) & (py < gt[:, 3])
        near = ((px - cx).abs() < self.radius * gt_stride) & ((py - cy).abs() < self.radius * gt_stride)
        on_level = stride == gt_stride                                      # (N,M)
        cand = inside & near & on_level
        missing = (~cand.any(0)).nonzero().squeeze(1)
        if missing.numel():                                                 # nearest-point fallback
            d2 = ((px - cx) ** 2 + (py - cy) ** 2).masked_fill(~on_level, float("inf"))
            cand[d2.argmin(0)[missing], missing] = True
        cost = wh.prod(1).unsqueeze(0).expand(N, M).masked_fill(~cand, float("inf"))
        pos = cand.any(1)
        return pos.nonzero().squeeze(1), cost.argmin(1)[pos]                # prefer smallest GT

    def _dfl(self, pred, target):
        tl = target.floor().long()
        tr = (tl + 1).clamp(max=self.reg_max - 1)
        wl = (tl + 1).to(target.dtype) - target
        return F.cross_entropy(pred, tl, reduction="none") * wl + F.cross_entropy(pred, tr, reduction="none") * (1 - wl)

    def forward(self, raw, targets):
        R = self.reg_max
        anchors, stride_t = make_anchors(raw, self.strides)
        pts = anchors * stride_t
        flat = torch.cat([r.flatten(2) for r in raw], 2)
        box_all = flat[:, : 4 * R].transpose(1, 2)                          # (B,N,4R)
        cls_all = flat[:, 4 * R:].transpose(1, 2)                           # (B,N,nc)
        l_box = l_cls = l_dfl = flat.new_zeros(())
        n_pos = 0
        for b, t in enumerate(targets):
            cls_t = torch.zeros_like(cls_all[b])
            if t.numel():
                pi, gi = self._assign(pts, stride_t, t[:, 1:5])
                tgt = t[gi]
                cls_t[pi, tgt[:, 0].long()] = 1.0
                dist = box_all[b, pi].reshape(-1, 4, R)
                ltrb = (dist.softmax(2) * self.proj).sum(2)                 # (Np,4) grid units
                s, ap = stride_t[pi], anchors[pi]
                pred = torch.cat([ap - ltrb[:, :2], ap + ltrb[:, 2:]], 1) * s
                l_box = l_box + generalized_box_iou_loss(pred, tgt[:, 1:5], reduction="sum")
                tb = tgt[:, 1:5] / s
                t_ltrb = torch.cat([ap - tb[:, :2], tb[:, 2:] - ap], 1).clamp(0, R - 1 - 0.01)
                l_dfl = l_dfl + self._dfl(dist.reshape(-1, R), t_ltrb.reshape(-1)).view(-1, 4).mean(1).sum()
                n_pos += pi.numel()
            l_cls = l_cls + sigmoid_focal_loss(cls_all[b], cls_t, reduction="sum")
        norm = max(n_pos, 1)
        box, cls, dfl = l_box / norm, l_cls / norm, l_dfl / norm
        return {"total": self.w_box * box + self.w_cls * cls + self.w_dfl * dfl, "box": box, "cls": cls, "dfl": dfl}


# ----------------------------------------------------------------------------------------
# Data: synthetic demo set + YOLO-format loader for real data (VisDrone / NWPU / AI-TOD)
# ----------------------------------------------------------------------------------------
class SyntheticBoxes(Dataset):
    """Noisy images with small coloured rectangles - just to exercise the pipeline."""

    PALETTE = torch.tensor([[0.9, 0.2, 0.2], [0.2, 0.9, 0.2], [0.2, 0.3, 0.9]])

    def __init__(self, n=64, size=256, nc=3, seed=0):
        self.n, self.size, self.nc, self.seed = n, size, nc, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.seed + i)
        ri = lambda lo, hi: int(torch.randint(lo, hi, (1,), generator=g))
        img = torch.rand(3, self.size, self.size, generator=g) * 0.25
        boxes = []
        for _ in range(ri(1, 5)):
            w, h = ri(8, 40), ri(8, 40)
            x1, y1, c = ri(0, self.size - w), ri(0, self.size - h), ri(0, self.nc)
            img[:, y1:y1 + h, x1:x1 + w] = self.PALETTE[c].view(3, 1, 1) * 0.8 + 0.1 * torch.rand(3, h, w, generator=g)
            boxes.append([c, x1, y1, x1 + w, y1 + h])
        return img, torch.tensor(boxes, dtype=torch.float32)


class YoloTxtDataset(Dataset):
    """Images + YOLO txt labels ('cls cx cy w h', normalised). Images are resized to imgsz x imgsz."""

    def __init__(self, img_dir, label_dir, imgsz=640):
        from pathlib import Path
        self.imgs = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        self.label_dir, self.imgsz = Path(label_dir), imgsz

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        from PIL import Image
        p, S = self.imgs[i], self.imgsz
        img = Image.open(p).convert("RGB").resize((S, S))
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
        lf = self.label_dir / f"{p.stem}.txt"
        rows = np.loadtxt(lf, ndmin=2) if lf.exists() and lf.stat().st_size else np.zeros((0, 5))
        r = torch.from_numpy(rows).float()
        t = torch.stack([r[:, 0], (r[:, 1] - r[:, 3] / 2) * S, (r[:, 2] - r[:, 4] / 2) * S,
                         (r[:, 1] + r[:, 3] / 2) * S, (r[:, 2] + r[:, 4] / 2) * S], 1) if len(r) else torch.zeros(0, 5)
        return x, t


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


# ----------------------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------------------
def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def sanity_checks():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 128, 128)
    assert CAM()(x).shape == x.shape
    assert ELLOM()(x).shape == x.shape
    v = VCHM(32)(torch.rand(2, 32, 40, 40))
    assert v.shape == (2, 63, 40, 40), v.shape                       # 2C-1 channels
    orsm = OrSM(16)
    f = torch.rand(2, 16, 40, 40)
    assert orsm(f).shape == f.shape
    assert torch.allclose(orsm.direction_mask(f).sum(1), torch.ones(2, 40, 40))   # exactly one direction
    assert torch.allclose(orsm.kernels().sum((1, 2, 3)), torch.zeros(16 * 8), atol=1e-5)  # zero-sum ESCK
    ffm = FFM(16).eval()
    z1, z2 = torch.rand(2, 16, 20, 20), torch.rand(2, 16, 20, 20)
    assert torch.allclose(ffm.injected(z1, z2), ffm.injected(z1, z2, explicit=True), atol=1e-5)
    model = BDNet(nc=3)
    raw = model(x)
    print("level shapes:", [tuple(r.shape) for r in raw], f"| params: {count_params(model):.2f}M")
    print("sanity checks passed")


def train_demo(epochs=5, nc=3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, loss_fn = BDNet(nc=nc).to(device), SimpleDetectionLoss(nc).to(device)
    dl = DataLoader(SyntheticBoxes(64, 256, nc), batch_size=8, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        for imgs, tg in dl:
            losses = loss_fn(model(imgs.to(device)), [t.to(device) for t in tg])
            opt.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        print(f"epoch {ep + 1}/{epochs}  " + "  ".join(f"{k}={v.item():.3f}" for k, v in losses.items()))
    model.eval()
    img, gt = SyntheticBoxes(4, 256, nc, seed=999)[0]
    det = model.predict(img[None].to(device), conf_thres=0.05)[0]
    print("ground truth (cls,x1,y1,x2,y2):\n", gt.int().tolist())
    print("detections (x1,y1,x2,y2,score,cls) top 5:\n", det[:5].cpu())


if __name__ == "__main__":
    sanity_checks()
    train_demo()
