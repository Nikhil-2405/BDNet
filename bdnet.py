"""
BDNet: Bio-Inspired Dual-Backbone Small Object Detection Network  (PyTorch re-implementation)

Paper: Guan et al., "BDNet: Bio-Inspired Dual-Backbone Small Object Detection Network", CVPR 2026.
This is an *unofficial* implementation written from the paper text (the authors' code and the
supplementary material were not available), so ambiguous details are marked "INTERPRETATION".

Install:   pip install torch torchvision numpy pillow
Run:       python bdnet_interactive.py

Interactive workflow:
  1. Train on a labelled dataset in YOLO text format.
  2. Run object detection on an image using a trained checkpoint.
  3. Run internal model sanity checks.
Important: this is an unofficial research re-implementation. Its simplified training loss
is not the official YOLO12/Ultralytics loss; benchmark scores from the paper are not guaranteed.

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
        self.img_dir = Path(img_dir).expanduser()
        self.label_dir = Path(label_dir).expanduser()
        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.img_dir}")
        if not self.label_dir.is_dir():
            raise FileNotFoundError(f"Label directory does not exist: {self.label_dir}")
        if imgsz < 64 or imgsz % 16 != 0:
            raise ValueError("Image size must be at least 64 and divisible by 16 (for example, 256 or 640).")
        self.imgs = sorted(p for p in self.img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not self.imgs:
            raise ValueError(f"No .jpg, .jpeg or .png images found in: {self.img_dir}")
        self.imgsz = imgsz

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        from PIL import Image
        p, S = self.imgs[i], self.imgsz
        img = Image.open(p).convert("RGB").resize((S, S))
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
        lf = self.label_dir / f"{p.stem}.txt"
        try:
            rows = np.loadtxt(lf, ndmin=2) if lf.exists() and lf.stat().st_size else np.zeros((0, 5))
        except Exception as exc:
            raise ValueError(f"Could not read label file {lf}: {exc}") from exc
        if rows.size and (rows.ndim != 2 or rows.shape[1] != 5):
            raise ValueError(f"Label file {lf} must have five columns: class_id cx cy width height")
        if rows.size:
            if not np.isfinite(rows).all():
                raise ValueError(f"Label file {lf} contains NaN or infinite values")
            if np.any(rows[:, 0] < 0) or np.any(rows[:, 1:] < 0) or np.any(rows[:, 1:] > 1):
                raise ValueError(f"Label values in {lf} must be normalized to the range [0, 1]")
            if np.any(rows[:, 3:] <= 0):
                raise ValueError(f"Bounding-box width and height must be positive in {lf}")
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




# ----------------------------------------------------------------------------------------
# Interactive training and image inference
# ----------------------------------------------------------------------------------------
def _prompt_text(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{label}{suffix}: ").strip()
    return value if value else (default or "")


def _prompt_int(label: str, default: int, minimum: int = 1) -> int:
    while True:
        raw = _prompt_text(label, str(default))
        try:
            value = int(raw)
            if value < minimum:
                raise ValueError
            return value
        except ValueError:
            print(f"Please enter a whole number greater than or equal to {minimum}.")


def _prompt_float(label: str, default: float, minimum: float, maximum: float) -> float:
    while True:
        raw = _prompt_text(label, str(default))
        try:
            value = float(raw)
            if not minimum <= value <= maximum:
                raise ValueError
            return value
        except ValueError:
            print(f"Please enter a number from {minimum} to {maximum}.")


def _prompt_path(label: str, must_exist: bool = True, is_dir: bool | None = None) -> str:
    from pathlib import Path
    while True:
        raw = input(f"{label}: ").strip().strip('"').strip("'")
        if not raw:
            print("A path is required.")
            continue
        path = Path(raw).expanduser()
        if must_exist and not path.exists():
            print(f"Path not found: {path}")
            continue
        if must_exist and is_dir is True and not path.is_dir():
            print("Please enter a directory path.")
            continue
        if must_exist and is_dir is False and not path.is_file():
            print("Please enter a file path.")
            continue
        if not must_exist and path.exists():
            answer = input(f"{path} already exists. Overwrite it? [y/N]: ").strip().lower()
            if answer not in {"y", "yes"}:
                continue
        return str(path)


def _select_device() -> torch.device:
    available = torch.cuda.is_available()
    default = "cuda" if available else "cpu"
    choice = _prompt_text("Compute device (auto/cpu/cuda)", "auto").lower()
    if choice in {"auto", ""}:
        choice = default
    if choice == "cuda" and not available:
        print("CUDA is not available in this PyTorch installation; using CPU instead.")
        choice = "cpu"
    if choice not in {"cpu", "cuda"}:
        print("Unrecognised device choice; using", default)
        choice = default
    device = torch.device(choice)
    print(f"Using device: {device}")
    return device


def _mean_losses(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "box": 0.0, "cls": 0.0, "dfl": 0.0}
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            targets = [target.to(device) for target in targets]
            losses = loss_fn(model(images), targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
            batches += 1
    if batches == 0:
        raise RuntimeError("The data loader produced no batches. Check the image directory.")
    return {key: value / batches for key, value in totals.items()}


def train_on_yolo_dataset():
    """Train BDNet from a user-supplied YOLO-format dataset and save a checkpoint."""
    from pathlib import Path

    print("\n=== TRAIN BDNET ON A LABELLED DATASET ===")
    print("Expected labels: one .txt file per image, with rows: class_id cx cy width height")
    print("The four box values must be normalized between 0 and 1.\n")

    class_text = input("Enter class names in label-ID order, separated by commas\n"
                       "Example: pedestrian,car,bicycle,bus: ").strip()
    class_names = [name.strip() for name in class_text.split(",") if name.strip()]
    if not class_names:
        raise ValueError("At least one class name is required.")
    nc = len(class_names)

    train_img = _prompt_path("Training images directory", is_dir=True)
    train_lbl = _prompt_path("Training labels directory", is_dir=True)
    val_img_raw = input("Validation images directory (press Enter to skip): ").strip().strip('"').strip("'")
    val_lbl_raw = input("Validation labels directory (press Enter to skip): ").strip().strip('"').strip("'")
    if bool(val_img_raw) != bool(val_lbl_raw):
        raise ValueError("Provide both validation image and label directories, or leave both blank.")

    imgsz = _prompt_int("Input image size (must be divisible by 16)", 256, 64)
    if imgsz % 16 != 0:
        raise ValueError("Image size must be divisible by 16, for example 256, 320, 512 or 640.")
    epochs = _prompt_int("Number of training epochs", 20, 1)
    batch_size = _prompt_int("Batch size", 4, 1)
    learning_rate = _prompt_float("Learning rate", 0.001, 1e-6, 1.0)
    output_path = _prompt_path("Checkpoint output path", must_exist=False)
    if Path(output_path).suffix.lower() not in {".pt", ".pth"}:
        output_path += ".pt"

    train_ds = YoloTxtDataset(train_img, train_lbl, imgsz=imgsz)
    val_ds = YoloTxtDataset(val_img_raw, val_lbl_raw, imgsz=imgsz) if val_img_raw else None

    # Validate all label class IDs before starting a potentially long training run.
    for dataset_name, dataset in (("training", train_ds), ("validation", val_ds)):
        if dataset is None:
            continue
        for i, image_path in enumerate(dataset.imgs):
            label_path = dataset.label_dir / f"{image_path.stem}.txt"
            if label_path.exists() and label_path.stat().st_size:
                rows = np.loadtxt(label_path, ndmin=2)
                if rows.size and (np.any(rows[:, 0] >= nc) or np.any(rows[:, 0] != np.floor(rows[:, 0]))):
                    raise ValueError(
                        f"Class IDs in {label_path} must be integers from 0 to {nc - 1}. "
                        f"Your class names are: {class_names}"
                    )
        print(f"{dataset_name.capitalize()} images found: {len(dataset)}")

    device = _select_device()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
        num_workers=0, pin_memory=(device.type == "cuda")
    ) if val_ds is not None else None

    model = BDNet(nc=nc).to(device)
    loss_fn = SimpleDetectionLoss(nc=nc).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_loss = float("inf")
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nClasses ({nc}): {class_names}")
    print(f"Model parameters: {count_params(model):.2f} million")
    print("Training started. Loss values are optimization indicators, not accuracy percentages.\n")

    for epoch in range(1, epochs + 1):
        train_stats = _mean_losses(model, train_loader, loss_fn, device, optimizer)
        if val_loader is not None:
            val_stats = _mean_losses(model, val_loader, loss_fn, device)
            selection_loss = val_stats["total"]
            val_text = (f"val_loss={val_stats['total']:.4f} "
                        f"(box={val_stats['box']:.4f}, cls={val_stats['cls']:.4f}, dfl={val_stats['dfl']:.4f})")
        else:
            selection_loss = train_stats["total"]
            val_text = "validation skipped"

        print(
            f"Epoch {epoch:>3}/{epochs} | train_loss={train_stats['total']:.4f} "
            f"(box={train_stats['box']:.4f}, cls={train_stats['cls']:.4f}, dfl={train_stats['dfl']:.4f}) | {val_text}"
        )

        # Keep the checkpoint with the lowest validation loss (or last training loss if no validation set).
        if selection_loss < best_loss:
            best_loss = selection_loss
            payload = {
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "nc": nc,
                "class_names": class_names,
                "imgsz": imgsz,
                "channels": (16, 32, 64, 128),
                "neck_ch": 64,
                "reg_max": 16,
                "best_selection_loss": best_loss,
                "epoch": epoch,
                "training_note": "Unofficial BDNet re-implementation with SimpleDetectionLoss; not official YOLO12 loss.",
            }
            torch.save(payload, output)
            print(f"  Saved best checkpoint: {output}")

    print("\nTraining finished.")
    print(f"Best selection loss: {best_loss:.4f}")
    print(f"Checkpoint: {output.resolve()}")
    print("Next step: choose image detection and load this checkpoint.")


def _load_checkpoint(checkpoint_path: str, device: torch.device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # compatibility with older PyTorch releases
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "This file is not a checkpoint saved by the interactive training option. "
            "Train the model first or provide a compatible checkpoint."
        )
    nc = int(checkpoint["nc"])
    class_names = checkpoint.get("class_names", [f"class_{i}" for i in range(nc)])
    imgsz = int(checkpoint.get("imgsz", 256))
    channels = tuple(checkpoint.get("channels", (16, 32, 64, 128)))
    neck_ch = int(checkpoint.get("neck_ch", 64))
    reg_max = int(checkpoint.get("reg_max", 16))
    model = BDNet(nc=nc, chs=channels, neck_ch=neck_ch, reg_max=reg_max).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, class_names, imgsz, checkpoint


def detect_in_user_image():
    """Run inference on one image, save an annotated image and a CSV file."""
    import csv
    from pathlib import Path
    from PIL import Image, ImageDraw, ImageFont

    print("\n=== DETECT OBJECTS IN AN IMAGE ===")
    checkpoint_path = _prompt_path("Trained BDNet checkpoint (.pt/.pth)", is_dir=False)
    image_path = _prompt_path("Input image path", is_dir=False)
    device = _select_device()
    model, class_names, imgsz, checkpoint = _load_checkpoint(checkpoint_path, device)

    conf = _prompt_float("Confidence threshold (0 to 1)", 0.25, 0.0, 1.0)
    iou = _prompt_float("IoU threshold for NMS (0 to 1)", 0.60, 0.0, 1.0)
    max_det = _prompt_int("Maximum detections to return", 100, 1)

    original = Image.open(image_path).convert("RGB")
    original_w, original_h = original.size
    resized = original.resize((imgsz, imgsz), Image.Resampling.BILINEAR)
    image_tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float().div(255.0)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        detections = model.predict(image_tensor, conf_thres=conf, iou_thres=iou, max_det=max_det)[0]

    annotated = original.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, original_w // 60))
    except OSError:
        font = ImageFont.load_default()

    stem = Path(image_path).stem
    output_image = Path(image_path).with_name(f"{stem}_bdnet_result.jpg")
    output_csv = Path(image_path).with_name(f"{stem}_bdnet_detections.csv")
    rows = []
    sx, sy = original_w / imgsz, original_h / imgsz

    for det in detections.detach().cpu().tolist():
        x1, y1, x2, y2, score, class_id_float = det
        class_id = int(class_id_float)
        # The model sees a stretched square image; map the predicted coordinates back per axis.
        x1 = min(max(x1 * sx, 0.0), float(original_w - 1))
        x2 = min(max(x2 * sx, 0.0), float(original_w - 1))
        y1 = min(max(y1 * sy, 0.0), float(original_h - 1))
        y2 = min(max(y2 * sy, 0.0), float(original_h - 1))
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        label = f"{class_name} {score:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=max(2, original_w // 400))
        text_box = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_box, fill=(255, 40, 40))
        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(float(score), 6),
            "x_min": round(x1, 2),
            "y_min": round(y1, 2),
            "x_max": round(x2, 2),
            "y_max": round(y2, 2),
        })

    annotated.save(output_image, quality=95)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["class_id", "class_name", "confidence", "x_min", "y_min", "x_max", "y_max"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== DETECTION SUMMARY ===")
    print(f"Input image: {image_path}")
    print(f"Image size: {original_w} x {original_h} pixels")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Confidence threshold: {conf:.2f} | NMS IoU threshold: {iou:.2f}")
    print(f"Objects detected: {len(rows)}")
    if rows:
        for index, row in enumerate(rows, start=1):
            print(
                f"{index}. {row['class_name']} | confidence={row['confidence']:.3f} | "
                f"box=({row['x_min']}, {row['y_min']}, {row['x_max']}, {row['y_max']})"
            )
    else:
        print("No object passed the selected confidence threshold.")
    print(f"\nAnnotated image saved to: {output_image}")
    print(f"Detection details saved to: {output_csv}")
    print("\nInterpretation: confidence is the model's score for a predicted class, not a guarantee of correctness.")


def run_sanity_checks():
    print("\nRunning BDNet module and tensor-shape checks...")
    sanity_checks()
    print("These checks verify code execution and tensor dimensions; they do not measure detection accuracy.")


def interactive_menu():
    print("=" * 72)
    print("BDNet - BIO-INSPIRED DUAL-BACKBONE SMALL OBJECT DETECTION")
    print("=" * 72)
    print("This program can train on your labelled dataset or detect objects using trained weights.")
    print("For real image detection, train the network first or provide a compatible checkpoint.")

    while True:
        print("\nChoose an operation:")
        print("  1. Train BDNet using a YOLO-format dataset")
        print("  2. Detect objects in an image using a trained checkpoint")
        print("  3. Run model sanity checks")
        print("  4. Exit")
        choice = input("Enter choice (1-4): ").strip()
        try:
            if choice == "1":
                train_on_yolo_dataset()
            elif choice == "2":
                detect_in_user_image()
            elif choice == "3":
                run_sanity_checks()
            elif choice == "4":
                print("Exiting BDNet. Goodbye.")
                break
            else:
                print("Please select 1, 2, 3 or 4.")
        except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
            print(f"\nOperation could not be completed: {exc}")
            print("Check the paths, dataset labels and settings, then try again.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")


if __name__ == "__main__":
    interactive_menu()
