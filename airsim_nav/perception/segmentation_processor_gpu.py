#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation Processor – Binary Obstacle Mask  (GPU‑accelerated) + Fixed ID Injection
────────────────────────────────────────────────────────────────────────────────────────
• Sky.* → 255 | Landscape.* → 0 ID ataması her çağrıda yapılır
• non_obstacle_ids desteği: geri kalan her şey engel kabul edilir (maskede beyaz)
"""

from __future__ import annotations
from pathlib import Path
import argparse, time, sys
from typing import List, Set, Tuple

try:
    import cupy as xp
    _GPU_BACKEND = "cupy"
except ModuleNotFoundError:
    import numpy as xp
    _GPU_BACKEND = "numpy"

import matplotlib.pyplot as plt
import airsim
from airsim_nav.config import Params

DEFAULT_IDS: Set[int] = {
    44, 83, 96, 104, 108, 110, 117, 128, 132, 147,
    166, 191, 198, 212, 109, 151, 170, 182, 46, 26,
    42, 232, 224, 174, 24, 172, 61, 197, 210, 211,
    36, 60, 126, 22, 6, 248, 156,
}

# ─────────────── ID Atama Fonksiyonu ───────────────
def _assign_fixed_ids(client: airsim.MultirotorClient):
    try:
        client.simSetSegmentationObjectID("Sky.*", 255, True)
        client.simSetSegmentationObjectID("Landscape.*", 0, True)

        #print("[seg] Sabit ID atamaları: Sky → 255, Landscape → 0")
    except Exception as e:
        print("[seg] ID atama başarısız:", e)

def _apply_orient(mask: "xp.ndarray", orient: dict | None) -> "xp.ndarray":
    if not orient:
        return mask
    rot = orient.get("rotate", 0) % 360
    if rot:
        k = rot // 90
        mask = xp.rot90(mask, k=k)
    if orient.get("flip_h"):
        mask = xp.fliplr(mask)
    if orient.get("flip_v"):
        mask = xp.flipud(mask)
    return mask

def _resolve_params(cam: str | None, vehicle: str | None,
                    target_ids: Set[int] | None,
                    params: Params) -> Tuple[str, str, Set[int]]:
    seg_cfg = params.segmentation or {}

    cam_out     = cam     or seg_cfg.get("cam",     params.seg_cam)
    vehicle_out = vehicle or seg_cfg.get("vehicle", params.vehicle)

    # non_obstacle_ids → engel = tüm 0-255 dışında kalanlar
    if "non_obstacle_ids" in seg_cfg:
        all_ids = set(range(256))
        non_obs = set(seg_cfg["non_obstacle_ids"])
        ids_out = all_ids - non_obs
        #if seg_cfg.get("verbose", True):
            #print(f"[seg] Engel olarak işaretlenen ID'ler: {sorted(ids_out)}")
    else:
        ids_out = (target_ids if target_ids is not None else
                   set(seg_cfg.get("target_ids", [])) or DEFAULT_IDS)

    return cam_out, vehicle_out, ids_out

def _build_lookup(ids: Set[int]) -> "xp.ndarray":
    lut = xp.zeros(256, dtype=xp.uint8)
    lut[xp.asarray(list(ids), dtype=xp.uint8)] = 1
    return lut

_LUT_CACHE: dict[frozenset[int], "xp.ndarray"] = {}

def fetch_mask(client: airsim.MultirotorClient,
               cam: str | None              = None,
               vehicle: str | None          = None,
               target_ids: Set[int] | None  = None,
               *,
               params: Params | None        = None) -> "xp.ndarray":

    _assign_fixed_ids(client)  # Gökyüzü ve zemin ID'lerini sabitle

    params = params or Params.load()
    cam, vehicle, target_ids = _resolve_params(cam, vehicle, target_ids, params)

    rsp = client.simGetImages([airsim.ImageRequest(
        cam, airsim.ImageType.Segmentation,
        pixels_as_float=False, compress=False)],
        vehicle_name=vehicle)[0]

    if rsp.width == 0 or rsp.height == 0:
        raise RuntimeError("Segmentation camera returned empty image.")

    w, h = rsp.width, rsp.height
    buf = (xp.frombuffer if _GPU_BACKEND == "cupy" else xp.frombuffer)(
        rsp.image_data_uint8, dtype=xp.uint8)
    bpp = buf.size // (w * h)

    if bpp == 1:
        ids = buf.reshape(h, w)
    else:
        img  = buf.reshape(h, w, bpp)
        chan = int(xp.argmax(img.sum(axis=(0, 1))).item())
        ids  = img[:, :, chan]

    key = frozenset(target_ids)
    lut = _LUT_CACHE.get(key)
    if lut is None:
        lut = _build_lookup(target_ids)
        _LUT_CACHE[key] = lut

    mask = lut[ids]  # 1 = engel (beyaz), 0 = boşluk

    # Kamera yönü dönüşü
    seg_cfg = (params.segmentation or {})
    orient = (seg_cfg.get("orient_map", {}).get(cam)
              or seg_cfg.get("orient"))
    if orient is None:
        if seg_cfg.get("verbose", True):
            print(f"[seg] ⚠ Kamera '{cam}' için orient tanımsız – dönüş uygulanmadı")
    else:
        mask = _apply_orient(mask, orient)

    return mask

# ───────────── CLI Arayüz ─────────────
def _cli() -> None:
    p = argparse.ArgumentParser(description="Segmentation mask viewer (GPU)")
    p.add_argument("--cam")
    p.add_argument("--vehicle")
    p.add_argument("--ids", type=int, nargs="*")
    p.add_argument("--view", action="store_true")
    p.add_argument("--save", metavar="PNG")
    p.add_argument("--cfg", metavar="YAML",
                   help="Farklı YAML dosyası (default=config.yaml")
    args = p.parse_args()

    cfg = Params.load(args.cfg)
    cam, veh, ids = _resolve_params(args.cam, args.vehicle,
                                    set(args.ids) if args.ids else None,
                                    cfg)

    cli = airsim.MultirotorClient(); cli.confirmConnection()
    backend = "GPU (CuPy)" if _GPU_BACKEND == "cupy" else "CPU (NumPy)"
    print(f"[seg] Connected  (vehicle='{veh}', cam='{cam}', backend={backend})")

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
                obs_pct = float(mask.mean()) * 100.0
                print(f"[seg] stream {fps:5.1f} Hz   obstacle_px={obs_pct:4.1f}%")
                t0, n = time.perf_counter(), 0

            if args.view or args.save:
                img = xp.asnumpy(mask) if _GPU_BACKEND == "cupy" else mask
                ax.clear(); ax.imshow(img, cmap="gray", interpolation="nearest")
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

if __name__ == "__main__":
    _cli()
