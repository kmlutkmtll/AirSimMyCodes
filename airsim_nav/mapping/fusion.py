#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lidar + Segmentation Fusion (rev-21 May 2025, v0.5)
────────────────────────────────────────────────────
* Seg-orient düzeltmesi varsayılan.
* Lidar noktalarına fazladan rotasyon kaldırıldı.
* `fusion.logic = and|union` (YAML) → AND / OR birleşimi seçilebilir.
"""
from __future__ import annotations

# ───────── STD LIB ─────────
import argparse, math, time, threading
from typing import Iterable, Set

# ───────── 3RD-PARTY ───────
import numpy as np
import matplotlib.pyplot as plt
import airsim
from airsim.utils import to_eularian_angles   # AirSim ≥1.8

try:
    import scipy.ndimage as ndi
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ───────── PROJE MODÜLLERİ ─────────
from airsim_nav.config import Params
from airsim_nav.perception.lidar_processor import occupancy_grid
from airsim_nav.perception.segmentation_processor import (
    fetch_mask, DEFAULT_IDS,
)

# ───────── KONFİG ─────────
cfg        = Params.load()
lidar_cfg  = cfg.lidar or {}
GRID: int  = lidar_cfg.get("grid_dim", cfg.grid_dim)
CELL: float= lidar_cfg.get("cell_size", cfg.cell_size)
VEH        = cfg.vehicle
LIDAR      = cfg.lidar_name

# ───────── PAYLAŞIMLI GRİD ─────────
latest_grid: np.ndarray | None = None
_grid_lock                = threading.Lock()

# ───────── YARDIMCILAR ─────────
def morphological_filter(grid: np.ndarray, open_size: int = 3) -> np.ndarray:
    """Basit binary aç-kapa (SciPy varsa)."""
    if not HAS_SCIPY:
        return grid
    selem  = np.ones((open_size, open_size), np.uint8)
    opened = ndi.binary_opening(grid.astype(bool), structure=selem)
    closed = ndi.binary_closing(opened,           structure=selem)
    return closed.astype(np.uint8)

# ───────── ANA DÖNGÜ ─────────
def _fusion_loop(*,
                 mode: str       = "fixed",      # fixed | scaled | depthcam
                 depth_m: float  = 20.0,
                 h_fov_deg: int  = 90,
                 ids: Set[int]   = DEFAULT_IDS,
                 view: bool      = False,
                 do_morph: bool  = False,
                 morph_size: int = 3) -> None:
    """Arka planda fusion – global *latest_grid* günceller."""
    global latest_grid

    cli = airsim.MultirotorClient(); cli.confirmConnection()
    print(f"[fusion] connected  (mode={mode}, grid={GRID}, cell={CELL:.2f})")

    H_FOV = math.radians(h_fov_deg)
    cx = cy = GRID // 2
    CAM_YAW_OFFSET = 0.0          # ek ofset kaldırıldı

    # === İzleme penceresi ===
    if view:
        dpi  = 96
        side = GRID / 28
        plt.rcParams["figure.figsize"] = (12, 4)
        fig, (axL, axS, axF) = plt.subplots(1, 3, figsize=(side*3, side),
                                            dpi=dpi, constrained_layout=True)
        imL = axL.imshow(np.zeros((GRID, GRID)), cmap="Reds",   vmin=0, vmax=1, origin="lower")
        imS = axS.imshow(np.zeros((GRID, GRID)), cmap="Greens", vmin=0, vmax=1, origin="lower")
        imF = axF.imshow(np.zeros((GRID, GRID)), cmap="gray_r", vmin=0, vmax=1, origin="lower")
        for ax, ttl in zip((axL, axS, axF), ("Lidar", "Seg Mask", "Fused")):
            ax.set_title(ttl); ax.set_xticks([]); ax.set_yticks([])

    # --- Maskeyi grid'e projekte eden fonksiyon ---
    def proj_to_grid(mask: np.ndarray,
                     depth_map: np.ndarray | None = None) -> np.ndarray:
        yaw = to_eularian_angles(
            cli.getMultirotorState(vehicle_name=VEH)
               .kinematics_estimated.orientation
        )[2]
        h, w = mask.shape
        yy, xx = np.nonzero(mask)
        if xx.size == 0:
            return np.zeros((GRID, GRID), np.uint8)

        # pixel → mesafe
        if mode == "fixed":
            depth = np.full_like(xx, depth_m, dtype=float)
        elif mode == "scaled":
            depth = np.interp(yy, (0, h), (depth_m * 3.0, depth_m * 0.5))
        else:  # depthcam
            if depth_map is None:
                return np.zeros((GRID, GRID), np.uint8)
            depth = depth_map[yy, xx]
            depth[(depth <= 0) | np.isnan(depth)] = depth_m

        # pixel → azimut
        az_cam = (xx / w - 0.5) * H_FOV
        az = az_cam + yaw + CAM_YAW_OFFSET
        gx = np.round(depth * np.cos(az) / CELL + cx).astype(int)
        gy = np.round(depth * np.sin(az) / CELL + cy).astype(int)

        seg_g = np.zeros((GRID, GRID), np.uint8)
        valid = (0 <= gx) & (gx < GRID) & (0 <= gy) & (gy < GRID)
        seg_g[gy[valid], gx[valid]] = 1
        return seg_g

    # === Döngü ===
    t0 = time.perf_counter(); frame = 0
    try:
        while True:
            # 1) Lidar
            ld = cli.getLidarData(lidar_name=LIDAR, vehicle_name=VEH)
            if len(ld.point_cloud) < 3:
                time.sleep(0.03); continue
            pts = np.asarray(ld.point_cloud, np.float32).reshape(-1, 3)
            pts = pts[np.isfinite(pts).all(axis=1)]      # NaN vs inf filtre

            # AirSim çıktısı body-frame; ekstra rotasyon yok
            lidar_g = occupancy_grid(pts, cfg)

            # 2) Segmentation mask (orient uygulanmış)
            mask = fetch_mask(cli, None, None, ids, params=cfg)

            # 3) Depth (opsiyonel)
            depth_img: np.ndarray | None = None
            if mode == "depthcam":
                req_cam = (cfg.segmentation or {}).get("cam", cfg.seg_cam)
                dr = cli.simGetImages([airsim.ImageRequest(
                    req_cam, airsim.ImageType.DepthPlanar,
                    pixels_as_float=True)], VEH)[0]
                dh, dw = dr.height, dr.width
                depth_raw = np.array(dr.image_data_float, np.float32).reshape(dh, dw)
                if (dh, dw) != mask.shape:
                    ry = math.ceil(mask.shape[0] / dh)
                    rx = math.ceil(mask.shape[1] / dw)
                    depth_img = np.repeat(np.repeat(depth_raw, ry, 0), rx, 1)[:mask.shape[0], :mask.shape[1]]
                else:
                    depth_img = depth_raw

            seg_g = proj_to_grid(mask, depth_img)

            # ---------- Lidar ∧/∨ Seg birleşimi ----------
            if (cfg.fusion or {}).get("logic", "and") == "and":
                fused_base = np.logical_and(lidar_g, seg_g).astype(np.uint8)
            else:           # logic == "union"
                fused_base = np.logical_or(lidar_g, seg_g).astype(np.uint8)

            fused = (morphological_filter(fused_base, morph_size)
                     if do_morph else fused_base)

            with _grid_lock:
                latest_grid = fused

            frame += 1
            if frame % 30 == 0:
                fps = frame / (time.perf_counter() - t0)
                print(f"[fusion] {fps:5.1f} Hz  (mode={mode})")
                t0, frame = time.perf_counter(), 0

            if view:
                imL.set_data(lidar_g); imS.set_data(seg_g); imF.set_data(fused)
                plt.pause(0.001)

    except KeyboardInterrupt:
        print("\n[fusion] user interrupt – thread exit")

# ───────── DIŞA AÇIK API ─────────
def start_fusion_thread(*,
                        view: bool = False,
                        mode: str = "fixed",
                        depth: float = 20.0,
                        h_fov: int = 90,
                        ids: Iterable[int] | None = None,
                        do_morph: bool = False,
                        morph_size: int = 3) -> None:
    ids_set = set(ids) if ids else DEFAULT_IDS
    threading.Thread(target=_fusion_loop,
                     kwargs=dict(view=view, mode=mode, depth_m=depth,
                                 h_fov_deg=h_fov, ids=ids_set,
                                 do_morph=do_morph, morph_size=morph_size),
                     daemon=True).start()
    print(f"[fusion] thread launched (mode={mode}, do_morph={do_morph})")

def get_latest_fused_grid(copy: bool = True) -> np.ndarray | None:
    with _grid_lock:
        if latest_grid is None:
            return None
        return latest_grid.copy() if copy else latest_grid

# ───────── CLI ─────────
def _cli() -> None:
    p = argparse.ArgumentParser(description="Lidar + Segmentation fusion viewer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--view", action="store_true")
    p.add_argument("--mode", choices=["fixed", "scaled", "depthcam"], default="fixed")
    p.add_argument("--depth", type=float, default=20.0)
    p.add_argument("--h_fov", type=float, default=120.0)
    p.add_argument("--ids", type=int, nargs="*")
    p.add_argument("--morph", action="store_true")
    p.add_argument("--morph_size", type=int, default=3)
    args = p.parse_args()

    _fusion_loop(view=args.view, mode=args.mode, depth_m=args.depth,
                 h_fov_deg=args.h_fov, ids=set(args.ids) if args.ids else DEFAULT_IDS,
                 do_morph=args.morph, morph_size=args.morph_size)

if __name__ == "__main__":
    _cli()
