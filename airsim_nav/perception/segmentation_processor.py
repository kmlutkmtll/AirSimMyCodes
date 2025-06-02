#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation Processor – Binary Obstacle Mask   (rev‑21 May 2025, v0.2)
────────────────────────────────────────────────────────────────────────
YENİ:
* Kamera adı, hedef ID listesi vb. artık `params.segmentation` bloğundan
  da okunabilir.  YAML’de örnek:
  ```yaml
  segmentation:
    cam: front_center
    target_ids: [44, 83, 96, 104]
  ```
* `fetch_mask()` → argüman verilmezse sırasıyla
  1) fonksiyon parametresi
  2) `params.segmentation.*`
  3) global default seçilir.
"""
from __future__ import annotations

from pathlib import Path
import argparse, time, sys
from typing import List, Set

import numpy as np
import matplotlib.pyplot as plt
import airsim

from airsim_nav.config import Params

# ─────────────────── Varsayılan ENGEL ID seti ──────────────
DEFAULT_IDS: Set[int] = {
    44, 83, 96, 104, 108, 110, 117, 128, 132, 147, 166,
    191, 198, 212, 109, 151, 170, 182, 46, 26, 42, 232, 224,
    174, 24, 172, 61, 197, 210, 211, 36, 60, 126, 22, 6, 248,
    156,
}

# ───────────────────── Yardımcı FONKSİYONLAR ──────────────────

def _apply_orient(mask: np.ndarray, orient: dict | None) -> np.ndarray:
    """rotate / flip mask according to orient dict.

    orient = {
        "rotate": 0|90|180|270,
        "flip_h": True/False,
        "flip_v": True/False,
    }
    """
    if not orient:
        return mask
    rot = orient.get("rotate", 0) % 360
    if rot:
        k = rot // 90
        mask = np.rot90(mask, k=k)
    if orient.get("flip_h"):
        mask = np.fliplr(mask)
    if orient.get("flip_v"):
        mask = np.flipud(mask)
    return mask


def _resolve_params(cam: str | None, vehicle: str | None,
                    target_ids: Set[int] | None,
                    params: Params) -> tuple[str, str, Set[int]]:
    """Args öncelik sırası: fonksiyon argümanı > YAML.segmentation > default."""
    seg_cfg = params.segmentation or {}

    cam_out     = cam        or seg_cfg.get("cam", params.seg_cam)
    vehicle_out = vehicle    or seg_cfg.get("vehicle", params.vehicle)
    ids_out     = (target_ids if target_ids is not None else
                   set(seg_cfg.get("target_ids", [])) or DEFAULT_IDS)
    return cam_out, vehicle_out, ids_out


def fetch_mask(client: airsim.MultirotorClient,
               cam: str | None              = None,
               vehicle: str | None          = None,
               target_ids: Set[int] | None  = None,
               *,
               params: Params | None        = None) -> np.ndarray:
    """AirSim simGetImages → 0/1 engel maskesi."""
    params = params or Params.load()
    cam, vehicle, target_ids = _resolve_params(cam, vehicle, target_ids, params)

    rsp = client.simGetImages([airsim.ImageRequest(
        cam, airsim.ImageType.Segmentation,
        pixels_as_float=False, compress=False)],
        vehicle_name=vehicle)[0]

    if rsp.width == 0 or rsp.height == 0:
        raise RuntimeError("Segmentation camera returned empty image.")

    w, h = rsp.width, rsp.height
    buf  = np.frombuffer(rsp.image_data_uint8, dtype=np.uint8)
    bpp  = buf.size // (w * h)            # byte per pixel (1,3,4)

    # --- Kanal seçimi ---
    if bpp == 1:                          # PF_SceneStencil (önerilen)
        ids = buf.reshape(h, w)
    else:                                 # RGB/RGBA stencil
        img  = buf.reshape(h, w, bpp)
        chan = int(np.argmax([img[:, :, k].sum() for k in range(bpp)]))
        ids  = img[:, :, chan]

    mask = np.isin(ids, list(target_ids)).astype(np.uint8)

    # YAML'deki orient_map veya orient ayarini uygula
    seg_cfg = params.segmentation or {}
    orient = seg_cfg.get("orient_map", {}).get(cam) or seg_cfg.get("orient")
    if orient is None:
        if seg_cfg.get("verbose", True):
            print(
                f"[seg] \N{WARNING SIGN} Kamera '{cam}' icin orient tanimsiz \N{EM DASH} donus uygulanmadi"
            )
    else:
        mask = _apply_orient(mask, orient)
    return mask

# ───────────────────────── CLI / TEST ──────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Segmentation mask visualiser / PNG saver")

    p.add_argument("--cam")
    p.add_argument("--vehicle")
    p.add_argument("--ids", type=int, nargs="*")
    p.add_argument("--view", action="store_true")
    p.add_argument("--save", metavar="PNG")
    args = p.parse_args()

    cfg = Params.load()
    cam, veh, ids = _resolve_params(args.cam, args.vehicle,
                                    set(args.ids) if args.ids else None,
                                    cfg)

    cli = airsim.MultirotorClient(); cli.confirmConnection()
    print(f"[seg] Connected  (vehicle='{veh}', cam='{cam}')")

    if args.view or args.save:
        plt.rcParams["figure.figsize"] = (6, 4)
        fig, ax = plt.subplots()

    t0 = time.perf_counter(); n = 0
    try:
        while True:
            mask = fetch_mask(cli, cam, veh, ids, params=cfg)
            n += 1
            if n % 30 == 0:
                fps = n / (time.perf_counter() - t0)
                print(f"[seg] stream {fps:5.1f} Hz   obstacle_px={mask.mean()*100:4.1f}%")
                t0, n = time.perf_counter(), 0

            if args.view or args.save:
                ax.clear(); ax.imshow(mask, cmap="gray", interpolation="nearest")
                ax.set_title("Segmentation Mask (white = obstacle)")
                ax.set_xticks([]); ax.set_yticks([])
                plt.pause(0.001)

            if args.save:
                out = Path(args.save)
                fig.savefig(out, dpi=120)
                print(f"[seg] Mask saved → {out}")
                sys.exit(0)
    except KeyboardInterrupt:
        print("\n[seg] user interrupt – exiting")

# --------------------------------------------------------------------
if __name__ == "__main__":
    _cli()
