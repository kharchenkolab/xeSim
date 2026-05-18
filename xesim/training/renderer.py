"""Train the v16 (PatchGAN) renderer.

This is the production renderer trainer. v16 = v13_gan architecture
(V37bUNet sigmoid + CellEncoder per-cell VAE + PatchGAN discriminator)
trained with two efficiency tricks:

* `combined_fwd`: one generator forward serves both D and G updates per
  tile instead of two separate forwards (~+8% wall).
* `d_skip = 2`: D update every 2 G steps (~+18% wall, GAN dynamics fine).

Loss recipe (from v10 → v13 lineage):

* Per-channel std-weighted Charbonnier reconstruction
* Sobel 3-channel gradient L1
* DINOv2 perceptual L2 (weight 0.05)
* Foreground radial log-power spectrum L1 (weight 0.10)
* Encoder KL (weight 0.01)
* PatchGAN hinge adversarial (weight 0.05, ramped over first 500 steps)

Checkpoint selection: val_dino + val_spec ("realism" metric).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cell_encoder import CROP_SIZE, CellEncoder, extract_cell_crops
from ..mechanistic_eval import DINOv2Embedder
from ..structural_refiner import V37bUNet, build_structural_channels
from ..torch_utils import get_device


# ---------------------------------------------------------------------------
# Loss helpers (copied from the v10 Tier-1 recipe)
# ---------------------------------------------------------------------------


def _gradient_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    def _grad(x):
        dx = x[..., :, 1:] - x[..., :, :-1]
        dy = x[..., 1:, :] - x[..., :-1, :]
        return dx, dy
    da_x, da_y = _grad(a); db_x, db_y = _grad(b)
    return (da_x - db_x).abs().mean() + (da_y - db_y).abs().mean()


def _charbonnier_per_channel(pred, target, mask, eps: float = 1e-3,
                                 *, weight_mode: str = "uniform",
                                 custom_weights: list[float] | None = None,
                                 ) -> torch.Tensor:
    """Per-channel Charbonnier-loss with configurable per-channel weighting.

    weight_mode:
      "inv_std" (default, v16/v19 behavior): ch_w = 1/std(target). Aggressively
        favors low-dynamic-range channels (e.g., aSMA), under-emphasizes wide-
        range channels (DAPI). Causes ~15-20% DAPI under-emission.
      "uniform": ch_w = 1. Balanced per-channel loss.
      "inv_sqrt_std": ch_w = 1/sqrt(std). Softer version of inv_std.
      "custom":   ch_w from custom_weights (caller-supplied list of length C).
        Values are normalized so the mean weight = 1, so absolute scale
        doesn't change the loss magnitude. Useful when inv_std isn't
        aggressive enough on a specific channel (e.g. aSMA's sparse high
        peaks need a stronger push than its std-based weight gives).
    """
    diff = torch.sqrt((pred - target) ** 2 + eps * eps)
    if weight_mode == "uniform":
        ch_w = torch.ones(pred.shape[1], device=pred.device).view(1, -1, 1, 1)
    elif weight_mode == "inv_sqrt_std":
        ch_std = torch.std(target, dim=(0, 2, 3)).clamp(min=1e-2)
        ch_w = (1.0 / torch.sqrt(ch_std))
        ch_w = (ch_w / ch_w.mean()).view(1, -1, 1, 1)
    elif weight_mode == "custom":
        if custom_weights is None or len(custom_weights) != pred.shape[1]:
            raise ValueError(
                f"recon_weight_mode='custom' requires custom_weights of "
                f"length {pred.shape[1]} (got "
                f"{None if custom_weights is None else len(custom_weights)})")
        w = torch.tensor(custom_weights, device=pred.device, dtype=pred.dtype)
        ch_w = (w / w.mean()).view(1, -1, 1, 1)
    else:                                                  # "inv_std"
        ch_std = torch.std(target, dim=(0, 2, 3)).clamp(min=1e-2)
        ch_w = (1.0 / ch_std)
        ch_w = (ch_w / ch_w.mean()).view(1, -1, 1, 1)
    weighted = diff * ch_w * mask
    return weighted.sum() / (mask.sum() * pred.shape[1] + 1e-6)


def _radial_log_power(patches: torch.Tensor, num_bins: int) -> torch.Tensor:
    n, c, h, w = patches.shape
    fy = torch.fft.fftfreq(h, device=patches.device)
    fx = torch.fft.rfftfreq(w, device=patches.device)
    fy2, fx2 = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(fy2 * fy2 + fx2 * fx2)
    r_max = float(r.max().item())
    bin_idx = ((r / (r_max + 1e-9)) * num_bins).long().clamp(max=num_bins - 1)
    counts = torch.bincount(bin_idx.flatten(), minlength=num_bins).float().clamp(min=1.0)
    fft = torch.fft.rfft2(patches, norm="ortho")
    power = fft.real * fft.real + fft.imag * fft.imag
    flat = power.reshape(n, c, -1)
    idx = bin_idx.flatten().reshape(1, 1, -1).expand(n, c, -1)
    out = torch.zeros(n, c, num_bins, device=patches.device, dtype=patches.dtype)
    out.scatter_add_(2, idx, flat)
    out = out / counts.reshape(1, 1, -1)
    return torch.log(out + 1e-8)


def _spectrum_l1(pred, target, fg_mask, num_bins: int = 24) -> torch.Tensor:
    m = fg_mask.clamp(0.0, 1.0)
    if m.dim() == 3: m = m.unsqueeze(1)
    pred_r = _radial_log_power(pred * m, num_bins)
    target_r = _radial_log_power(target * m, num_bins)
    return torch.mean(torch.abs(pred_r.mean(dim=0) - target_r.mean(dim=0)))


# ---------------------------------------------------------------------------
# PatchGAN discriminator (single-scale, spectral-normed)
# ---------------------------------------------------------------------------


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, hidden: int = 32):
        super().__init__()
        h = hidden
        def block(ic, oc):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(ic, oc, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.net = nn.Sequential(
            block(in_channels, h),       # 256→128
            block(h, h * 2),             # 128→64
            block(h * 2, h * 4),         # 64→32
            block(h * 4, h * 4),         # 32→16
            nn.utils.spectral_norm(nn.Conv2d(h * 4, 1, 1)),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Vectorized GPU mask augmentation (preserves cell shape diversity)
# ---------------------------------------------------------------------------


def _centroids_grid(base_masks: torch.Tensor):
    K = base_masks.shape[-1]
    ys = torch.arange(K, device=base_masks.device, dtype=base_masks.dtype)
    ys_grid = ys.view(1, K, 1).expand(base_masks.shape[0], K, K)
    xs_grid = ys.view(1, 1, K).expand(base_masks.shape[0], K, K)
    wsum = base_masks.sum(dim=(1, 2)).clamp(min=1.0)
    cy = (base_masks * ys_grid).sum(dim=(1, 2)) / wsum
    cx = (base_masks * xs_grid).sum(dim=(1, 2)) / wsum
    return cy.long(), cx.long()


def _perturb_masks(base_masks: torch.Tensor, prob: float, rng: np.random.Generator):
    N, K, _ = base_masks.shape
    if prob <= 0 or N == 0: return base_masks
    do_aug = torch.from_numpy((rng.random(N) < prob).astype(np.bool_)).to(base_masks.device)
    if not bool(do_aug.any()): return base_masks
    modes = torch.from_numpy(rng.integers(0, 4, size=N)).to(base_masks.device)
    erode_iters = int(rng.integers(1, 4)); dilate_iters = int(rng.integers(1, 3))
    halfcut_axis = int(rng.integers(0, 2)); halfcut_side = int(rng.integers(0, 2))
    out = base_masks.clone()
    bm = base_masks.unsqueeze(1)
    ero = bm
    for _ in range(erode_iters):
        ero = -F.max_pool2d(-ero, 3, stride=1, padding=1)
    ero = ero.squeeze(1)
    dil = bm
    for _ in range(dilate_iters):
        dil = F.max_pool2d(dil, 3, stride=1, padding=1)
    dil = dil.squeeze(1) * base_masks
    cy, cx = _centroids_grid(base_masks)
    rows = torch.arange(K, device=base_masks.device).view(1, K, 1).expand(N, K, K)
    cols = torch.arange(K, device=base_masks.device).view(1, 1, K).expand(N, K, K)
    if halfcut_axis == 0:
        cy_b = cy.view(N, 1, 1).expand(N, K, K)
        keep = (rows >= cy_b) if halfcut_side else (rows < cy_b)
    else:
        cx_b = cx.view(N, 1, 1).expand(N, K, K)
        keep = (cols >= cx_b) if halfcut_side else (cols < cx_b)
    cut = base_masks * keep.to(base_masks.dtype)
    sel_e = do_aug & (modes == 0); sel_d = do_aug & (modes == 1); sel_c = do_aug & (modes == 2)
    if sel_e.any(): out[sel_e] = ero[sel_e]
    if sel_d.any(): out[sel_d] = dil[sel_d]
    if sel_c.any(): out[sel_c] = cut[sel_c]
    return out


# ---------------------------------------------------------------------------
# Tile loading
# ---------------------------------------------------------------------------


def _load_tile(canonical_dir: Path, crop: dict, device, n_type_one_hot: int,
                  cid_to_type: dict, n_channels: int | None = None,
                  per_type_channel_means: np.ndarray | None = None,
                  stromal_type_indices: tuple[int, ...] | None = None):
    with np.load(canonical_dir / crop["npz_path"], allow_pickle=True) as d:
        all_real = np.asarray(d["images"])
        # Use first n_channels if specified; otherwise all available channels
        if n_channels is None:
            n_channels = all_real.shape[0]
        real = all_real[:n_channels, :256, :256].astype(np.float32)
        cell_label = np.asarray(d["cell_label"])[:256, :256].astype(np.int32)
        nucleus_label = np.asarray(d["nucleus_label"])[:256, :256].astype(np.int32)
        cell_ids = [str(v) for v in d["cell_ids"].tolist()] if "cell_ids" in d.files else []
    nz = np.unique(cell_label); nz = nz[nz > 0]
    nuc_remap = np.zeros_like(cell_label, dtype=np.int32)
    for lbl in nz:
        cp = cell_label == int(lbl); nuc_in = nucleus_label[cp]; nuc_in = nuc_in[nuc_in > 0]
        if nuc_in.size:
            best = int(np.bincount(nuc_in).argmax())
            pix = (nucleus_label == best) & cp
            if pix.sum() > 5: nuc_remap[pix] = int(lbl)
    compact = np.zeros_like(cell_label, dtype=np.int32)
    for new_idx, old_lbl in enumerate(nz, start=1):
        compact[cell_label == int(old_lbl)] = new_idx
        nuc_remap[nuc_remap == int(old_lbl)] = new_idx
    cti = {}; per_cell_type = []
    for new_idx, old_lbl in enumerate(nz, start=1):
        cid = cell_ids[new_idx - 1] if new_idx - 1 < len(cell_ids) else None
        t = int(cid_to_type.get(str(cid), 0)) if cid else 0
        cti[new_idx] = t; per_cell_type.append(t)
    struct_np, _ = build_structural_channels(
        compact, nuc_remap, cti, n_type_one_hot,
        per_type_channel_means=per_type_channel_means,
        stromal_type_indices=stromal_type_indices,
    )
    cell_crops, base_masks = extract_cell_crops(real, compact)
    return {
        "real_t": torch.from_numpy(real).unsqueeze(0).to(device),
        "compact_t": torch.from_numpy(compact).long().to(device),
        "struct_t": torch.from_numpy(struct_np).unsqueeze(0).to(device),
        "cell_crops_t": torch.from_numpy(cell_crops).to(device),
        "base_masks_t": torch.from_numpy(base_masks).to(device),
        "types_t": torch.from_numpy(np.array(per_cell_type, dtype=np.int64)).to(device),
        "n_cells": int(len(nz)),
    }


# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------


def train_renderer(
    canonical_dir: Path,
    out_path: Path,
    *,
    steps: int = 8000,
    batch_tiles: int = 2,
    latent_dim: int = 12,
    hidden_v37b: int = 96,
    hidden_d: int = 32,
    lr: float = 2e-4,
    beta_kl: float = 0.01,
    mask_aug_prob: float = 0.5,
    grad_3ch_weight: float = 1.0,
    dino_weight: float = 0.05,
    spectrum_weight: float = 0.10,
    spectrum_bins: int = 24,
    charb_eps: float = 1e-3,
    adv_weight: float = 0.05,
    adv_warmup_steps: int = 500,
    d_skip: int = 2,
    combined_fwd: bool = True,
    log_every: int = 200,
    device: str | None = None,
    n_channels: int = 3,
    channel_names: list[str] | None = None,
    renderer_variant: str = "v17",
    use_per_type_means: bool = False,
    stromal_type_indices: tuple[int, ...] | None = None,
    recon_weight_mode: str = "uniform",
    recon_weight_custom: list[float] | None = None,
) -> Path:
    """Train v16 renderer (V37bUNet + CellEncoder + PatchGAN) from canonical crops.

    Writes a single checkpoint at `out_path` containing
    `{v37b, encoder, disc, val_dino, val_spec, best_score, …}` and a
    sibling `training.log` next to it.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_dir = Path(canonical_dir)
    device = get_device(device)
    torch.set_float32_matmul_precision("high")
    try: torch.backends.cudnn.allow_tf32 = True
    except Exception: pass

    manifest = json.loads((canonical_dir / "manifest.json").read_text())
    ct = json.loads((canonical_dir / "cell_types.json").read_text())
    type_names = ct["type_names"]
    cid_to_type = {}
    for crop in ct.get("crops", []):
        for cid, idx in crop.get("cell_id_to_type_index", {}).items():
            cid_to_type[str(cid)] = int(idx)
    n_type_one_hot = len(type_names) - 1

    # Phase 2.D optional conditioning: per-pixel expected-intensity + stromal density
    per_type_means: np.ndarray | None = None
    if use_per_type_means:
        ptm_path = canonical_dir / "per_type_channel_means.npy"
        if not ptm_path.exists():
            raise FileNotFoundError(
                f"use_per_type_means=True but {ptm_path} missing. "
                f"Run scripts/phase2d_fit_per_type_means.py first.")
        per_type_means = np.load(ptm_path).astype(np.float32)
        print(f"[v20] loaded per_type_channel_means {per_type_means.shape}")
    if stromal_type_indices:
        print(f"[v20] stromal_type_indices = {stromal_type_indices}")
    n_extra = (per_type_means.shape[1] if per_type_means is not None else 0) \
              + (1 if stromal_type_indices else 0)
    n_struct = 8 + n_type_one_hot + n_extra
    if n_extra:
        print(f"[v20] n_struct = 8 + {n_type_one_hot} + {n_extra} = {n_struct}")
    v37b = V37bUNet(in_channels=n_struct + latent_dim, hidden=hidden_v37b,
                       out_channels=n_channels).to(device)
    encoder = CellEncoder(n_types=n_type_one_hot, latent_dim=latent_dim,
                              in_channels=n_channels + 1).to(device)
    disc = PatchDiscriminator(in_channels=n_channels, hidden=hidden_d).to(device)
    print(f"V37b: {sum(p.numel() for p in v37b.parameters()):,}; "
          f"Encoder: {sum(p.numel() for p in encoder.parameters()):,}; "
          f"Disc: {sum(p.numel() for p in disc.parameters()):,}")

    opt_g = torch.optim.AdamW(list(v37b.parameters()) + list(encoder.parameters()),
                                lr=lr, betas=(0.5, 0.9))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=lr, betas=(0.5, 0.9))

    train_crops = [c for c in manifest["crops"] if c.get("split") == "train"]
    val_crops = [c for c in manifest["crops"] if c.get("split") in ("val", "test")][:8]
    print(f"Loading {len(train_crops)} train + {len(val_crops)} val tiles ({n_channels} channels)...")
    train_tiles = [_load_tile(canonical_dir, c, device, n_type_one_hot, cid_to_type,
                                n_channels=n_channels,
                                per_type_channel_means=per_type_means,
                                stromal_type_indices=stromal_type_indices)
                    for c in train_crops]
    val_tiles = [_load_tile(canonical_dir, c, device, n_type_one_hot, cid_to_type,
                              n_channels=n_channels,
                              per_type_channel_means=per_type_means,
                              stromal_type_indices=stromal_type_indices)
                  for c in val_crops]

    # DINOv2 expects RGB (3 channels). For N>3 channels, take the first 3
    # (DAPI + membrane + 18S — the original Xenium tri-stain set). For N<3,
    # DINOv2 perceptual loss will be disabled.
    def _dino_input(x: torch.Tensor) -> torch.Tensor:
        return x[:, :3] if x.shape[1] >= 3 else x

    dino_usable = (n_channels >= 3 and dino_weight > 0)
    dino = DINOv2Embedder(device=device) if dino_usable else None
    if dino is not None:
        dino._ensure_loaded()
    with torch.no_grad():
        for tile in train_tiles + val_tiles:
            real_t = tile["real_t"]
            cell_mask = (tile["compact_t"] > 0).float().unsqueeze(0).unsqueeze(0)
            if dino is not None:
                f_r = dino.features_for_training(_dino_input(real_t))
                f_r_n = f_r / (f_r.norm(dim=-1, keepdim=True) + 1e-9)
                tile["dino_real_norm"] = f_r_n.detach()
            else:
                tile["dino_real_norm"] = None
            tile["cell_mask"] = cell_mask
            target_fg = real_t * cell_mask
            tile["spec_real"] = _radial_log_power(target_fg, spectrum_bins).mean(dim=0).detach()
    if dino is not None:
        print(f"DINOv2 features cached on first {min(3, n_channels)} of {n_channels} channels.")
    else:
        print(f"DINOv2 perceptual loss disabled (n_channels={n_channels}, dino_weight={dino_weight}).")

    rng_aug = np.random.default_rng(123)

    def forward_recon(tile, aug_prob):
        base_masks = tile["base_masks_t"]
        aug_masks = _perturb_masks(base_masks, aug_prob, rng_aug)
        crops_t = tile["cell_crops_t"] * aug_masks.unsqueeze(1)
        x_enc = torch.cat([crops_t, aug_masks.unsqueeze(1)], dim=1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mu, logsig = encoder(x_enc, tile["types_t"])
            mu = mu.float(); logsig = logsig.float().clamp(-4, 2)
            eps = torch.randn_like(mu)
            z = mu + torch.exp(logsig) * eps
            lut = torch.cat([torch.zeros(1, latent_dim, device=device, dtype=z.dtype), z], dim=0)
            lat_block = lut[tile["compact_t"]].permute(2, 0, 1).unsqueeze(0)
            cond = torch.cat([tile["struct_t"], lat_block], dim=1)
            recon = v37b(cond).float()
        return recon, mu, logsig

    def g_losses(recon, mu, logsig, tile):
        real_t = tile["real_t"]; cm = tile["cell_mask"]
        recon_loss = _charbonnier_per_channel(recon, real_t, cm, eps=charb_eps,
                                                  weight_mode=recon_weight_mode,
                                                  custom_weights=recon_weight_custom)
        grad_3ch = _gradient_l1(recon, real_t) if grad_3ch_weight > 0 else torch.zeros((), device=device)
        kl = -0.5 * (1 + 2*logsig - mu**2 - torch.exp(2*logsig)).sum(dim=1).mean()
        if dino is not None and tile["dino_real_norm"] is not None:
            f_p = dino.features_for_training(_dino_input(recon))
            f_pn = f_p / (f_p.norm(dim=-1, keepdim=True) + 1e-9)
            dino_perc = ((f_pn - tile["dino_real_norm"]) ** 2).sum(dim=-1).mean()
        else:
            dino_perc = torch.zeros((), device=device)
        if spectrum_weight > 0:
            pred_radial = _radial_log_power(recon * cm, spectrum_bins).mean(dim=0)
            spec = torch.mean(torch.abs(pred_radial - tile["spec_real"]))
        else:
            spec = torch.zeros((), device=device)
        adv_g = -disc(recon).mean()
        return recon_loss, grad_3ch, kl, dino_perc, spec, adv_g

    best_score = float("inf"); best_state = None
    log_f = open(out_path.parent / "training.log", "w")
    acc = {k: torch.zeros((), device=device) for k in ("r", "g", "k", "d", "s", "adv", "dloss")}
    acc_n = 0

    for step in range(steps):
        adv_w = min(1.0, step / max(adv_warmup_steps, 1)) * adv_weight
        do_d = (step % max(d_skip, 1)) == 0
        idx = np.random.randint(0, len(train_tiles), batch_tiles)

        if combined_fwd:
            opt_g.zero_grad()
            if do_d: opt_d.zero_grad()
            for ti in idx:
                tile = train_tiles[ti]
                recon, mu, logsig = forward_recon(tile, mask_aug_prob)
                if do_d:
                    d_real = disc(tile["real_t"]); d_fake = disc(recon.detach())
                    d_loss = (torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()) * 0.5
                    d_loss.backward()
                    with torch.no_grad(): acc["dloss"] += d_loss.detach()
                rl, gl, kl, dp, sp, ag = g_losses(recon, mu, logsig, tile)
                total = rl + grad_3ch_weight*gl + dino_weight*dp + spectrum_weight*sp + beta_kl*kl + adv_w*ag
                total.backward()
                with torch.no_grad():
                    acc["r"] += rl.detach(); acc["g"] += gl.detach(); acc["k"] += kl.detach()
                    acc["d"] += dp.detach(); acc["s"] += sp.detach(); acc["adv"] += ag.detach()
                    acc_n += 1
            if do_d: opt_d.step()
            opt_g.step()
        else:
            if do_d:
                opt_d.zero_grad()
                for ti in idx:
                    tile = train_tiles[ti]
                    with torch.no_grad():
                        recon, _, _ = forward_recon(tile, mask_aug_prob)
                    d_real = disc(tile["real_t"]); d_fake = disc(recon.detach())
                    d_loss = (torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()) * 0.5
                    d_loss.backward()
                    with torch.no_grad(): acc["dloss"] += d_loss.detach()
                opt_d.step()
            opt_g.zero_grad()
            for ti in idx:
                tile = train_tiles[ti]
                recon, mu, logsig = forward_recon(tile, mask_aug_prob)
                rl, gl, kl, dp, sp, ag = g_losses(recon, mu, logsig, tile)
                total = rl + grad_3ch_weight*gl + dino_weight*dp + spectrum_weight*sp + beta_kl*kl + adv_w*ag
                total.backward()
                with torch.no_grad():
                    acc["r"] += rl.detach(); acc["g"] += gl.detach(); acc["k"] += kl.detach()
                    acc["d"] += dp.detach(); acc["s"] += sp.detach(); acc["adv"] += ag.detach()
                    acc_n += 1
            opt_g.step()

        if step % log_every == 0 or step == steps - 1:
            n = max(acc_n, 1)
            vals = {k: float(v.item()) / n for k, v in acc.items()}
            for v in acc.values(): v.zero_()
            acc_n = 0
            with torch.no_grad():
                v_dino = v_spec = 0.0
                for tile in val_tiles:
                    recon, _, _ = forward_recon(tile, 0.0)
                    if dino is not None and tile["dino_real_norm"] is not None:
                        f_p = dino.features_for_training(_dino_input(recon))
                        f_pn = f_p / (f_p.norm(dim=-1, keepdim=True) + 1e-9)
                        v_dino += float(((f_pn - tile["dino_real_norm"]) ** 2).sum(dim=-1).mean())
                    v_spec += float(_spectrum_l1(recon, tile["real_t"], tile["cell_mask"], spectrum_bins))
                v_dino /= max(len(val_tiles), 1)
                v_spec /= max(len(val_tiles), 1)
            score = v_dino + v_spec
            line = (f"step {step:5d}  recon {vals['r']:.4f}  dino {vals['d']:.4f}  "
                    f"spec {vals['s']:.4f}  adv {vals['adv']:.4f}  d_loss {vals['dloss']:.4f}  "
                    f"adv_w {adv_w:.3f}  KL {vals['k']:.4f}  "
                    f"val_dino {v_dino:.4f}  val_spec {v_spec:.4f}  score {score:.4f}")
            print(line); log_f.write(line + "\n"); log_f.flush()
            if score < best_score:
                best_score = score
                best_state = {
                    "v37b": {k: v.cpu().clone() for k, v in v37b.state_dict().items()},
                    "encoder": {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                    "disc": {k: v.cpu().clone() for k, v in disc.state_dict().items()},
                    "step": step, "val_dino": v_dino, "val_spec": v_spec,
                }
    log_f.close()
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint (no eval steps?)")
    torch.save({**best_state, "best_score": best_score,
                "latent_dim": latent_dim, "hidden_v37b": hidden_v37b,
                "hidden_d": hidden_d, "n_type_one_hot": n_type_one_hot,
                "n_channels": int(n_channels),
                "channel_names": list(channel_names) if channel_names else None,
                "renderer_variant": str(renderer_variant),
                "use_per_type_means": bool(use_per_type_means),
                "stromal_type_indices": (list(stromal_type_indices)
                                          if stromal_type_indices else None),
                "n_struct": int(n_struct)},
                out_path)
    print(f"Saved best (score={best_score:.4f}) → {out_path}")
    return out_path
