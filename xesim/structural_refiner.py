"""Plan3 v37 structural-only refiner pipeline.

Two-stage rendering:
  Stage 1 (v37b): deterministic U-Net mapping (structural conditioning
                  + per-cell latent) -> faithful smooth base image.
  Stage 2 (v37h): EDM residual diffusion on top of v37b, generating
                  high-freq texture detail without relocating features.

A single entry point `render_scene(scene, ...)` takes a MechanisticScene
(populated with `latent_vector` per cell, or None to draw fresh latents)
and returns a (3, H, W) float32 image in [0, 1].

Both stages share the same 15-channel structural conditioning derived
from the scene's geometry + cell-type one-hot + per-cell latent + a
high-freq noise channel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt

from .mechanistic_scene import MechanisticScene
from .torch_utils import get_device


PER_CELL_LATENT_DIM = 4

# EDM-residual schedule constants (matches scripts/v37h_resdiff.py)
RES_SIGMA_DATA = 0.05
RES_SIGMA_MIN = 0.001
RES_SIGMA_MAX = 0.5
RES_RHO = 7.0


# ---------- Structural conditioning channels ----------

def build_structural_channels(
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
    cell_type_indices: dict[int, int],
    n_type_one_hot: int,
    *,
    per_type_channel_means: np.ndarray | None = None,
    stromal_type_indices: tuple[int, ...] | None = None,
    stromal_sigma_px: float = 12.0,
) -> tuple[np.ndarray, list[str]]:
    """Build deterministic structural conditioning channels.

    Returns ``(channels: (C, H, W) float32, names)``. Channel count:
        ``8 + n_type_one_hot + (n_channels if per_type_channel_means else 0)
              + (1 if stromal_type_indices else 0)``.

    Base channels (always present):
      0  cell_mask
      1  nucleus_mask
      2  cell_boundary  (1-px ring at the cell edge)
      3  nucleus_boundary
      4  intercellular_edge  (cell pixels adjacent to a *different* cell)
      5  cell_interior_distance   (distance from boundary, inside cells, /30 clipped)
      6  cell_exterior_distance   (distance from boundary, outside cells, /30 clipped)
      7  nucleus_interior_distance(distance from nucleus boundary, /15 clipped)
      8..7+T  per-type one-hot (each channel is the union of cells of one type)

    Optional Phase 2.D channels:
      ``per_type_channel_means`` (shape ``(T+1, n_channels)``): for each cell,
        emit one channel per stain holding ``per_type_channel_means[type, c]``
        on the cell's pixels (0 elsewhere). Direct per-pixel expected-intensity
        prior — strongest aSMA-cell-type fix.
      ``stromal_type_indices``: extracellular channel = Gaussian-smoothed
        union of cells with these type indices, evaluated OUTSIDE cell_mask.
        Gives the renderer a spatial slot for fiber signal in stromal regions.

    `cell_type_indices` maps compacted cell label (1..N) -> type index in
    [1..T]; cells with type 0 (unknown) are excluded from the one-hot.
    """
    h, w = cell_label.shape
    chans: list[np.ndarray] = []
    names: list[str] = []

    cell_mask = (cell_label > 0).astype(np.float32)
    nuc_mask = (nucleus_label > 0).astype(np.float32)
    chans.append(cell_mask); names.append("cell_mask")
    chans.append(nuc_mask); names.append("nucleus_mask")

    cell_bnd = (cell_mask - binary_erosion(cell_mask).astype(np.float32)).astype(np.float32)
    nuc_bnd = (nuc_mask - binary_erosion(nuc_mask).astype(np.float32)).astype(np.float32)
    chans.append(cell_bnd); names.append("cell_boundary")
    chans.append(nuc_bnd); names.append("nucleus_boundary")

    # intercellular_edge: cell pixels whose neighbor is a different positive cell
    pad = np.pad(cell_label, 1, mode="constant", constant_values=0)
    diff = np.zeros((h, w), dtype=bool)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        sh = pad[1+dy:1+dy+h, 1+dx:1+dx+w]
        diff |= (cell_label > 0) & (sh > 0) & (cell_label != sh)
    chans.append(diff.astype(np.float32))
    names.append("intercellular_edge")

    # Distance fields, normalized by 30 px (~6 µm at 0.21 µm/px).
    cell_in_dist = (distance_transform_edt(cell_mask) / 30.0).clip(0, 2.0).astype(np.float32)
    cell_out_dist = (distance_transform_edt(1 - cell_mask) / 30.0).clip(0, 2.0).astype(np.float32)
    nuc_in_dist = (distance_transform_edt(nuc_mask) / 15.0).clip(0, 2.0).astype(np.float32)
    chans.extend([cell_in_dist, cell_out_dist, nuc_in_dist])
    names.extend(["cell_interior_distance", "cell_exterior_distance", "nucleus_interior_distance"])

    # Per-type one-hot (excluding the unknown / type-0 channel).
    type_oh = np.zeros((n_type_one_hot, h, w), dtype=np.float32)
    for label_val, type_idx in cell_type_indices.items():
        if type_idx <= 0 or type_idx > n_type_one_hot:
            continue
        type_oh[type_idx - 1, cell_label == int(label_val)] = 1.0
    for ti in range(n_type_one_hot):
        chans.append(type_oh[ti])
        names.append(f"cell_type_{ti+1}")

    # Optional Phase 2.D: per-pixel expected-intensity prior.
    if per_type_channel_means is not None:
        ptcm = np.asarray(per_type_channel_means, dtype=np.float32)
        # Expect shape (n_type_one_hot+1, n_channels) — row 0 = unknown
        n_emit = ptcm.shape[1]
        expected = np.zeros((n_emit, h, w), dtype=np.float32)
        for label_val, type_idx in cell_type_indices.items():
            row = ptcm[type_idx] if 0 <= type_idx < ptcm.shape[0] else ptcm[0]
            m = (cell_label == int(label_val))
            for c in range(n_emit):
                expected[c, m] = float(row[c])
        for c in range(n_emit):
            chans.append(expected[c])
            names.append(f"expected_intensity_{c}")

    # Optional Phase 2.D: extracellular stromal density (for fiber stains).
    if stromal_type_indices:
        stromal_mask = np.zeros((h, w), dtype=np.float32)
        for label_val, type_idx in cell_type_indices.items():
            if int(type_idx) in stromal_type_indices:
                stromal_mask[cell_label == int(label_val)] = 1.0
        # Gaussian blur over the union of stromal-cell masks, evaluated
        # everywhere; this gives a continuous "stromal-tissue proximity" map
        # that is nonzero in extracellular regions adjacent to stromal cells.
        from scipy.ndimage import gaussian_filter
        stromal_smooth = gaussian_filter(stromal_mask, sigma=float(stromal_sigma_px))
        # Normalize so a tightly-packed stromal region peaks near 1.0
        peak = stromal_smooth.max()
        if peak > 1e-6:
            stromal_smooth = stromal_smooth / peak
        chans.append(stromal_smooth.astype(np.float32))
        names.append("stromal_density")

    return np.stack(chans, axis=0), names


def inject_per_cell_latents(
    struct_chans: torch.Tensor,
    cell_labels: torch.Tensor,
    latent_vectors: dict[int, np.ndarray] | None = None,
    latent_dim: int = PER_CELL_LATENT_DIM,
) -> torch.Tensor:
    """Append per-cell latent channels.

    cell_labels: (B, H, W) int — compacted labels 1..N per tile.
    latent_vectors: optional dict[label -> (latent_dim,) array] supplying a
        specific latent per cell; otherwise random N(0, I) is drawn.

    Returns (B, C + latent_dim, H, W).
    """
    B, C, H, W = struct_chans.shape
    device = struct_chans.device
    labels_np = cell_labels.detach().cpu().numpy() if hasattr(cell_labels, "detach") else cell_labels
    lat_blocks = [torch.zeros((B, H, W), dtype=struct_chans.dtype, device=device)
                  for _ in range(latent_dim)]
    for b in range(B):
        labels = labels_np[b]
        unique = np.unique(labels); unique = unique[unique > 0]
        if len(unique) == 0:
            continue
        max_lbl = int(labels.max())
        # Use a SCENE-DEPENDENT but per-cell-deterministic random draw
        # for cells without supplied latents. Hash the cell label to seed
        # a small RNG so the same cell always gets the same fallback
        # latent, regardless of how many other cells are in the scene.
        # The old code used `np.random.randn()` per cell, which produced
        # latents whose values depended on the iteration ORDER — dropping
        # one cell shifted every subsequent cell's random draw, causing
        # spurious render changes across the whole tile.
        for li in range(latent_dim):
            lut = torch.zeros(max_lbl + 1, dtype=struct_chans.dtype, device=device)
            for lbl in unique:
                if latent_vectors is not None and int(lbl) in latent_vectors:
                    val = float(latent_vectors[int(lbl)][li])
                else:
                    # Deterministic per (cell label, latent dim) — no
                    # dependence on iteration order.
                    rng = np.random.default_rng(hash((int(lbl), li)) & 0xFFFFFFFF)
                    val = float(rng.standard_normal())
                lut[int(lbl)] = val
            label_t = torch.from_numpy(labels.astype(np.int64)).to(device)
            lat_blocks[li][b] = lut[label_t]
    return torch.cat([struct_chans, *[b.unsqueeze(1) for b in lat_blocks]], dim=1)


def inject_noise_channel(struct_chans: torch.Tensor, noise_std: float = 0.3) -> torch.Tensor:
    """Append a high-freq Gaussian-noise channel, masked to cell pixels."""
    B, _, H, W = struct_chans.shape
    cell_mask = (struct_chans[:, 0:1] > 0.5).float()
    noise = noise_std * torch.randn(B, 1, H, W, device=struct_chans.device,
                                    dtype=struct_chans.dtype) * cell_mask
    return torch.cat([struct_chans, noise], dim=1)


# ---------- v37b base U-Net ----------

class V37bUNet(nn.Module):
    """Deterministic structural refiner — matches scripts/v37_prototype.StructUNet."""
    def __init__(self, in_channels: int, hidden: int = 64, out_channels: int = 3):
        super().__init__()
        self._out_channels = int(out_channels)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden, hidden*2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(hidden*2, hidden*2, 3, padding=1), nn.GELU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(hidden*2, hidden*4, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(hidden*4, hidden*4, 3, padding=1), nn.GELU(),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(hidden*4, hidden*4, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden*4, hidden*4, 3, padding=1), nn.GELU(),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(hidden*8, hidden*2, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden*2, hidden*2, 3, padding=1), nn.GELU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(hidden*4, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden*2, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.out = nn.Conv2d(hidden, self._out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        m = self.mid(e3)
        u3 = F.interpolate(m, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([e3, u3], dim=1))
        u2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([e2, u2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([e1, u1], dim=1))
        return torch.sigmoid(self.out(d1))


# ---------- v37h residual diffusion U-Net ----------

class _ConvBlock(nn.Module):
    """Matches scripts/v37f_edm.ConvBlock — state dict keys '<name>.net.X.weight'."""
    def __init__(self, in_c, out_c):
        super().__init__()
        gn2 = max(1, min(8, out_c // 4))
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.GroupNorm(gn2, out_c), nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.GroupNorm(gn2, out_c), nn.SiLU(),
        )
    def forward(self, x): return self.net(x)


class V37hUNet(nn.Module):
    """EDM residual diffusion U-Net — matches scripts/v37f_edm.EDMUNet."""
    def __init__(self, image_channels: int = 3, cond_channels: int = 23,
                 hidden: int = 48, highfreq_head: bool = True,
                 highfreq_scale: float = 1.0):
        super().__init__()
        self.highfreq_scale = float(highfreq_scale)
        in_c = image_channels + cond_channels + 1  # +1 for sigma map
        h = hidden
        self.enc1 = _ConvBlock(in_c, h)
        self.enc2 = _ConvBlock(h, h*2)
        self.enc3 = _ConvBlock(h*2, h*4)
        self.mid = _ConvBlock(h*4, h*4)
        self.dec3 = _ConvBlock(h*8, h*2)
        self.dec2 = _ConvBlock(h*4, h)
        self.dec1 = _ConvBlock(h*2, h)
        self.highfreq_head = highfreq_head
        if highfreq_head:
            self.out_low = nn.Conv2d(h, image_channels, 1)
            self.out_high = nn.Conv2d(h, image_channels, 1)
        else:
            self.out = nn.Conv2d(h, image_channels, 1)

    def forward(self, x, cond, c_noise):
        if c_noise.ndim == 1:
            c_noise = c_noise.view(-1, 1, 1, 1)
        c_noise_map = c_noise.expand(-1, 1, x.shape[-2], x.shape[-1])
        inp = torch.cat([x, cond, c_noise_map], dim=1)
        e1 = self.enc1(inp)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        m = self.mid(e3)
        u3 = F.interpolate(m, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        if self.highfreq_head:
            low = self.out_low(d1)
            high_raw = self.out_high(d1)
            high = high_raw - F.avg_pool2d(high_raw, 3, stride=1, padding=1)
            return low + self.highfreq_scale * high
        return self.out(d1)


def _edm_res_scales(sigma):
    s2 = sigma * sigma; sd2 = RES_SIGMA_DATA ** 2
    c_skip = sd2 / (s2 + sd2)
    c_out = sigma * RES_SIGMA_DATA / torch.sqrt(s2 + sd2)
    c_in = 1.0 / torch.sqrt(s2 + sd2)
    c_noise = 0.25 * torch.log(sigma)
    return c_skip, c_out, c_in, c_noise


def _edm_res_denoise(model, x_noisy, sigma, cond):
    c_skip, c_out, c_in, c_noise = _edm_res_scales(sigma)
    while c_skip.ndim < x_noisy.ndim:
        c_skip = c_skip.unsqueeze(-1); c_out = c_out.unsqueeze(-1); c_in = c_in.unsqueeze(-1)
    F_x = model(c_in * x_noisy, cond, c_noise)
    return c_skip * x_noisy + c_out * F_x


def _res_sampling_sigmas(n_steps: int, device):
    idx = torch.arange(n_steps, dtype=torch.float32, device=device)
    sigmas = (RES_SIGMA_MAX ** (1/RES_RHO) + idx / (n_steps - 1) *
              (RES_SIGMA_MIN ** (1/RES_RHO) - RES_SIGMA_MAX ** (1/RES_RHO))) ** RES_RHO
    return torch.cat([sigmas, torch.zeros(1, device=device, dtype=sigmas.dtype)])


@torch.no_grad()
def _heun_sample_residual(model, cond, n_steps: int, device, image_shape=(3, 256, 256)):
    B = cond.shape[0]
    sigmas = _res_sampling_sigmas(n_steps, device)
    x = torch.randn(B, *image_shape, device=device) * sigmas[0]
    for i in range(n_steps):
        s_cur, s_next = sigmas[i], sigmas[i+1]
        denoised = _edm_res_denoise(model, x, s_cur.expand(B), cond)
        d = (x - denoised) / s_cur
        x_next = x + (s_next - s_cur) * d
        if s_next > 0:
            denoised_next = _edm_res_denoise(model, x_next, s_next.expand(B), cond)
            d_next = (x_next - denoised_next) / s_next
            x_next = x + (s_next - s_cur) * 0.5 * (d + d_next)
        x = x_next
    return x


# ---------- Pipeline API ----------

@dataclass
class RefinerPipeline:
    """v37b + v37h structural refiner pipeline, loaded and ready to render."""
    v37b: V37bUNet
    v37h: V37hUNet | None
    device: torch.device
    n_type_one_hot: int
    shape_extra_chans: int = 0  # v6+: extra per-pixel shape-descriptor channels

    @classmethod
    def load(cls, v37b_ckpt: Path, v37h_ckpt: Path | None = None,
             device_name: str | None = None, n_type_one_hot: int = 7) -> "RefinerPipeline":
        device = get_device(device_name)
        # Infer v37b input channels from checkpoint
        ckpt_b = torch.load(v37b_ckpt, map_location=device, weights_only=False)
        # Two ckpt formats supported:
        #   (a) {"state_dict": {...}} from v37 production training
        #   (b) {"v37b": {...}, "encoder": {...}, ...} from v37b-VAE joint training
        if "state_dict" in ckpt_b:
            sd = ckpt_b["state_dict"]
        elif "v37b" in ckpt_b:
            sd = ckpt_b["v37b"]
        else:
            sd = ckpt_b
        # v6+: per-pixel shape-descriptor structural channels
        shape_extra = int(ckpt_b.get("shape_extra_chans", 0)) if isinstance(ckpt_b, dict) else 0
        first_weight = sd["enc1.0.weight"]
        in_c_b = int(first_weight.shape[1])
        hidden_b = int(first_weight.shape[0])
        v37b = V37bUNet(in_channels=in_c_b, hidden=hidden_b).to(device)
        v37b.load_state_dict(sd)
        v37b.eval()
        v37h = None
        if v37h_ckpt is not None:
            ckpt_h = torch.load(v37h_ckpt, map_location=device, weights_only=False)
            first_h = ckpt_h["state_dict"]["enc1.net.0.weight"]
            in_c_h = int(first_h.shape[1])
            hidden_h = int(first_h.shape[0])
            # in_c_h = 3 (image) + cond + 1 (sigma)
            cond_h = in_c_h - 3 - 1
            v37h = V37hUNet(image_channels=3, cond_channels=cond_h, hidden=hidden_h,
                            highfreq_head=True, highfreq_scale=1.0).to(device)
            v37h.load_state_dict(ckpt_h["state_dict"])
            v37h.eval()
        return cls(v37b=v37b, v37h=v37h, device=device,
                   n_type_one_hot=n_type_one_hot, shape_extra_chans=shape_extra)

    @torch.no_grad()
    def render_scene(self, scene: MechanisticScene,
                     type_names: list[str] | None = None,
                     latent_vectors: dict[int, np.ndarray] | None = None,
                     detail_alpha: float = 1.0,
                     n_residual_steps: int = 15,
                     seed: int | None = None) -> np.ndarray:
        """Render a MechanisticScene -> (3, H, W) float32 image in [0, 1].

        Per-cell latents: when the scene's cells have `latent_vector` set,
        those are used. Otherwise random latents are drawn at render time.
        Pass `latent_vectors={label: vec}` to override.

        `detail_alpha=0` skips the v37h residual stage (faster, smoother).
        `detail_alpha>0` blends the v37h residual: `final = v37b_out + α * resid`.
        """
        if seed is not None:
            torch.manual_seed(int(seed)); np.random.seed(int(seed))

        # Build label-to-type-index map from scene cells.
        if type_names is None:
            type_names = ["unknown"] + [f"type_{i}" for i in range(1, self.n_type_one_hot + 1)]
        type_to_idx = {name: i for i, name in enumerate(type_names)}
        # Compacify cell labels to 1..N (scene may have arbitrary integer labels)
        nz = np.unique(scene.cell_label); nz = nz[nz > 0]
        compact = np.zeros_like(scene.cell_label, dtype=np.int32)
        for new_idx, old_lbl in enumerate(nz, start=1):
            compact[scene.cell_label == int(old_lbl)] = new_idx
        # Per-cell type index (compact label -> type index)
        cell_type_indices: dict[int, int] = {}
        latent_lookup: dict[int, np.ndarray] | None = None
        if latent_vectors is None:
            latent_lookup = {}
        for new_idx, old_lbl in enumerate(nz, start=1):
            cell = next((c for c in scene.cells if int(c.label) == int(old_lbl)), None)
            if cell is None:
                cell_type_indices[new_idx] = 0
            else:
                type_idx = type_to_idx.get(cell.cell_type or "unknown", 0)
                cell_type_indices[new_idx] = type_idx
                if cell.latent_vector is not None and latent_vectors is None:
                    latent_lookup[new_idx] = np.asarray(cell.latent_vector, dtype=np.float32)
        # Remap nucleus labels: scene's nucleus_label uses old IDs that match cell IDs
        # (synth scenes) or arbitrary IDs (real). For real, build a remap.
        nucleus_remap = np.zeros_like(scene.nucleus_label, dtype=np.int32)
        for new_idx, old_lbl in enumerate(nz, start=1):
            # First try direct match: in synth scenes nucleus_label uses cell_label
            mask_direct = (scene.nucleus_label == int(old_lbl)) & (scene.cell_label == int(old_lbl))
            if mask_direct.any():
                nucleus_remap[mask_direct] = new_idx
            else:
                # Real-data fallback: find nucleus pixels inside this cell
                cell_pix = scene.cell_label == int(old_lbl)
                nuc_in = scene.nucleus_label[cell_pix]; nuc_in = nuc_in[nuc_in > 0]
                if nuc_in.size:
                    best = int(np.bincount(nuc_in).argmax())
                    nuc_pix = (scene.nucleus_label == best) & cell_pix
                    if nuc_pix.sum() > 5:
                        nucleus_remap[nuc_pix] = new_idx

        # Build structural conditioning
        struct_np, _names = build_structural_channels(
            compact, nucleus_remap, cell_type_indices, n_type_one_hot=self.n_type_one_hot,
        )
        # v6+: insert per-pixel shape-descriptor maps between the 8 base
        # channels and the per-type one-hot, matching the training-time
        # channel layout.
        if self.shape_extra_chans > 0:
            from xesim.nucleus_shape_descriptor import cell_shape_descriptors
            desc = cell_shape_descriptors(compact, nucleus_remap)
            H_, W_ = compact.shape
            shape_maps = np.zeros((self.shape_extra_chans, H_, W_), dtype=np.float32)
            for L, vec in desc.items():
                m = compact == int(L)
                if not m.any():
                    continue
                for i in range(min(self.shape_extra_chans, len(vec))):
                    shape_maps[i][m] = float(vec[i])
            struct_np = np.concatenate([struct_np[:8], shape_maps, struct_np[8:]], axis=0)
        struct_t = torch.from_numpy(struct_np).unsqueeze(0).to(self.device)  # (1, C, H, W)
        compact_t = torch.from_numpy(compact).unsqueeze(0).to(self.device)
        # BUG FIX: caller-supplied latent_vectors is keyed by ORIGINAL cell
        # label, but inject_per_cell_latents looks up by COMPACT label
        # (1..N). Remap to compact keys so latents land on the right cell.
        # Previously, lookup always missed -> every cell got a sequential
        # random latent, and dropping one cell shifted the order so the
        # entire tile rerendered. This was the source of cross-tile
        # instability in the single-cell-drop diagnostic.
        if latent_vectors is not None:
            remapped: dict[int, np.ndarray] = {}
            for new_idx, old_lbl in enumerate(nz, start=1):
                if int(old_lbl) in latent_vectors:
                    remapped[int(new_idx)] = latent_vectors[int(old_lbl)]
            eff_latents = remapped
        else:
            eff_latents = latent_lookup
        # Infer latent_dim from the v37b's first-conv input channels: total input
        # = n_structural + latent_dim. Allows checkpoints with non-default
        # latent_dim (e.g., v37b-VAE-v2 uses 12) to work without an arg.
        in_c = self.v37b.enc1[0].weight.shape[1]
        n_struct = 8 + self.shape_extra_chans + self.n_type_one_hot
        latent_dim = max(0, in_c - n_struct)
        struct_with_lat = inject_per_cell_latents(struct_t, compact_t, eff_latents,
                                                  latent_dim=latent_dim)

        # Stage 1: v37b deterministic base
        v37b_out = self.v37b(struct_with_lat)  # (1, 3, H, W) in [0, 1]

        if self.v37h is None or detail_alpha == 0.0:
            return v37b_out.squeeze(0).cpu().numpy()

        # Stage 2: v37h residual diffusion
        cond_full = torch.cat([v37b_out, struct_with_lat], dim=1)
        cond_full = inject_noise_channel(cond_full, noise_std=0.3)
        resid = _heun_sample_residual(self.v37h, cond_full, n_residual_steps, self.device)
        final = (v37b_out + detail_alpha * resid).clamp(0, 1)
        return final.squeeze(0).cpu().numpy()


