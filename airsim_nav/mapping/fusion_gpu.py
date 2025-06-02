#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lidar + Segmentation Fusion – Full GPU Edition (rev‑27 May 2025, v0.8‑cuda)
────────────────────────────────────────────────────────────────────────────
• Tüm compute yolunun %100 GPU üzerinde çalıştığı birleşik Occupancy Grid.
• Hem Lidar (CuPy) hem Segmentation projeksiyonu (CuPy) → tek dizi.
• Morphology (opsiyonel) → cupyx.scipy.ndimage (GPU) / scipy fallback.
• CLI & daemon thread API eskiyle birebir uyumlu.

Public API
----------
start_fusion_thread(view=False, mode="fixed"|"scaled"|"depthcam", …)
get_latest_fused_grid(copy=True)  → np.ndarray | None  (CPU kopyası)
"""
from __future__ import annotations

# ───────────────────── Runtime backend select ─────────────────────────────
try:
    import cupy as xp
    import cupyx.scipy.ndimage as xndi
    _GPU = True
except ModuleNotFoundError:
    import numpy as xp                      # type: ignore
    import scipy.ndimage as xndi            # type: ignore
    _GPU = False

import argparse, math, threading, time, sys
from typing import Iterable, Set

import matplotlib.pyplot as plt             # CPU‑side preview only
import airsim
from airsim.utils import to_eularian_angles

from airsim_nav.config import Params
from airsim_nav.perception.lidar_processor_gpu import occupancy_grid
from airsim_nav.perception.segmentation_processor_gpu import (
    fetch_mask, DEFAULT_IDS,
)




# ─────────────────── Config & shared state ────────────────────────────────
_cfg          = Params.load()
_lidar_cfg    = _cfg.lidar or {}
GRID   : int  = _lidar_cfg.get("grid_dim", _cfg.grid_dim)
CELL   : float= _lidar_cfg.get("cell_size", _cfg.cell_size)
VEH    : str  = _cfg.vehicle
LIDAR  : str  = _cfg.lidar_name

_latest_grid : xp.ndarray | None = None
_lock         = threading.Lock()



# ────────────────── Utilities ─────────────────────────────────────────────

def _morphological_filter(grid: xp.ndarray, size: int = 3) -> xp.ndarray:
    """Binary opening → closing to suppress speckles (GPU accelerated)."""
    if size <= 1:
        return grid
    se = xp.ones((size, size), xp.uint8)
    opened = xndi.binary_opening(grid.astype(bool), structure=se)
    closed = xndi.binary_closing(opened,        structure=se)
    return closed.astype(xp.uint8)


def _project_mask(mask: xp.ndarray, *, yaw: float, mode: str,
                  depth_m: float, depth_map: xp.ndarray | None,
                  h_fov: float) -> xp.ndarray:
    """Pixel mask (H×W) → occupancy grid (GRID×GRID) on GPU/CPU."""
    h, w = mask.shape
    yy, xx = xp.nonzero(mask)
    if xx.size == 0:
        return xp.zeros((GRID, GRID), xp.uint8)

    # ---------- 1) Depth per pixel -----------------------------------
    if mode == "fixed":
        depth = xp.full(xx.shape, depth_m, xp.float32)

    elif mode == "scaled":
        y_norm = yy.astype(xp.float32) / (h - 1)
        try:        # CuPy ≥12
            depth = xp.interp(y_norm, xp.array([0, 1]),
                              xp.array([depth_m * 3.0, depth_m * 0.5]))
        except AttributeError:
            depth = depth_m * (3.0 - 2.5 * y_norm)   # eşdeğer lineer

    else:  # "depthcam"
        if depth_map is None:
            return xp.zeros((GRID, GRID), xp.uint8)
        depth = depth_map[yy, xx]
        depth = xp.where((depth <= 0) | xp.isnan(depth), depth_m, depth)

    # ---------- 2) Azimuth per pixel (pinhole + yaw + offset) --------
    fx     = (w * 0.5) / math.tan(h_fov * 0.5)         # pinhole odak (px)
    cx_pix = (w - 1) * 0.5
    az_cam = xp.arctan((xx.astype(xp.float32) - cx_pix) / fx)
    # az = -(az_cam + yaw) - math.pi         #           + math.pi * 1.5     sabit –90 ° ofset
    az = az_cam

    # ---------- 3) Pixel projection → GRID --------------------------
    cx = cy = GRID // 2
    gx = xp.floor(depth * xp.cos(az) / CELL + cx).astype(xp.int32)
    gy = xp.floor(depth * xp.sin(az) / CELL + cy).astype(xp.int32)

    seg_g = xp.zeros((GRID, GRID), xp.uint8)
    valid = (gx >= 0) & (gx < GRID) & (gy >= 0) & (gy < GRID)
    seg_g[gy[valid], gx[valid]] = 1
    return seg_g


# ──────────────────── Core loop ───────────────────────────────────────────

def _fusion_loop(*, mode: str, depth_m: float, h_fov_deg: float,
                 ids: Set[int], view: bool, do_morph: bool,
                 morph_size: int) -> None:
    global _latest_grid

    cli = airsim.MultirotorClient(); cli.confirmConnection()
    print(f"[fusion] connected (GRID={GRID}, cell={CELL:.2f}, backend={'GPU' if _GPU else 'CPU'})")

    h_fov = math.radians(h_fov_deg)
    cx = cy = GRID // 2  # noqa – kept for readability

    # Preview window --------------------------------------------------------
    if view:
        plt.rcParams["figure.figsize"] = (12, 4)
        fig, (axL, axS, axF) = plt.subplots(1, 3, constrained_layout=True)
        ims = [ax.imshow(xp.asnumpy(xp.zeros((GRID, GRID))), cmap=cmap, vmin=0, vmax=1, origin="lower")
                for ax, cmap in zip((axL, axS, axF), ("Reds", "Greens", "gray_r"))]
        for ax, ttl in zip((axL, axS, axF), ("Lidar", "Seg", "Fused")):
            ax.set_title(ttl); ax.set_xticks([]); ax.set_yticks([])

        # ──────────────── ★ ÖN YÖN OKU ★ ─────────────────
        head_len = 0.08 * GRID  # ok uzunluğu ≈ grid’in %12’si
        arrow = None
        def _imshow(i: int, arr: xp.ndarray):
            ims[i].set_data(xp.asnumpy(arr))

        def _update_arrow(yaw_rad: float):
            nonlocal arrow
            if arrow is not None:
                arrow.remove()
            dx = head_len * math.cos(yaw_rad)
            dy = -head_len * math.sin(yaw_rad)  # matplotlib y-eksen ters çevrilmez
            arrow = axF.arrow(cx, cy, dx, dy,
                              head_width=0.02 * GRID,
                              head_length=0.04 * GRID,
                              fc='r', ec='r', linewidth=2)

    t0 = time.perf_counter(); frame = 0
    try:
        while True:
            # ----- 1) Lidar → occupancy grid (already GPU) ------------------
            ld = cli.getLidarData(lidar_name=LIDAR, vehicle_name=VEH)
            if len(ld.point_cloud) < 3:
                time.sleep(0.02); continue
            pts = xp.asarray(ld.point_cloud, xp.float32).reshape(-1, 3)
            pts = pts[xp.isfinite(pts).all(axis=1)]
            lidar_g = occupancy_grid(pts, _cfg)  # GPU array

            # ----- 2) Segmentation mask (GPU uint8) ------------------------
            mask = fetch_mask(cli, None, None, ids, params=_cfg)

            # DepthCam fetch if needed --------------------------------------
            depth_map = None
            if mode == "depthcam":
                req_cam = (_cfg.segmentation or {}).get("cam", _cfg.seg_cam)
                dr = cli.simGetImages([airsim.ImageRequest(req_cam, airsim.ImageType.DepthPlanar,
                                                             pixels_as_float=True)], VEH)[0]
                dh, dw = dr.height, dr.width
                depth_raw = xp.asarray(dr.image_data_float, xp.float32).reshape(dh, dw)
                # --- (1) Bilinear yeniden ölçekleme --------------

                import cv2
                if (dh, dw) != mask.shape:
                    depth_cpu = xp.asnumpy(depth_raw) if _GPU else depth_raw
                    depth_cpu = cv2.resize(
                        depth_cpu.astype("float32"),
                        (mask.shape[1], mask.shape[0]),  # (W, H)
                        interpolation=cv2.INTER_LINEAR
                    )
                    depth_map = xp.asarray(depth_cpu) if _GPU else depth_cpu
                else:
                    depth_map = depth_raw

                # --- (2) Depth_map'e de aynı orientasyonu uygula -----------
                from airsim_nav.perception.segmentation_processor_gpu import _apply_orient
                seg_cfg = (_cfg.segmentation or {})
                orient = (seg_cfg.get("orient_map", {}).get(
                    seg_cfg.get("cam", _cfg.seg_cam))
                          or seg_cfg.get("orient"))
                if orient is not None:
                    depth_map = _apply_orient(depth_map, orient)

            # Camera yaw -----------------------------------------------------
            yaw = to_eularian_angles(
                cli.getMultirotorState(vehicle_name=VEH).kinematics_estimated.orientation
            )[2]

            seg_g = _project_mask(mask, yaw=yaw, mode=mode, depth_m=depth_m,
                                  depth_map=depth_map, h_fov=h_fov)

            # ----- 3) Fusion logic -----------------------------------------
            if (_cfg.fusion or {}).get("logic", "union") in ("and", "intersect"):
                fused_base = xp.logical_and(lidar_g, seg_g)
            else:
                fused_base = xp.logical_or(lidar_g, seg_g)
            fused_base = fused_base.astype(xp.uint8)

            fused = (_morphological_filter(fused_base, morph_size)
                     if do_morph else fused_base)

            with _lock:
                _latest_grid = fused

            # Logging --------------------------------------------------------
            frame += 1
            if frame % 30 == 0:
                fps = frame / (time.perf_counter() - t0)
                print(f"[fusion] {fps:5.1f} Hz (mode={mode}, backend={'GPU' if _GPU else 'CPU'})")
                t0, frame = time.perf_counter(), 0

            # Preview -------------------------------------------------------
            if view:
                _imshow(0, lidar_g); _imshow(1, seg_g); _imshow(2, fused)
                yaw = to_eularian_angles(
                    cli.getMultirotorState(vehicle_name=VEH
                                           ).kinematics_estimated.orientation)[2]
                _update_arrow(yaw)
                plt.pause(0.001)
    except KeyboardInterrupt:
        print("\n[fusion] user interrupt – exiting")
        sys.exit(0)


# ─────────────────── Public threaded API ─────────────────────────────────--

def start_fusion_thread(*, view: bool = False, mode: str = "fixed",
                        depth: float = 20.0, h_fov: float = 90.0,
                        ids: Iterable[int] | None = None,
                        do_morph: bool = False, morph_size: int = 3) -> None:
    """Launch background fusion thread (daemon)."""
    thr = threading.Thread(target=_fusion_loop, daemon=True,
                           kwargs=dict(view=view, mode=mode, depth_m=depth,
                                       h_fov_deg=h_fov, ids=set(ids or DEFAULT_IDS),
                                       do_morph=do_morph, morph_size=morph_size))
    thr.start()
    print(f"[fusion] thread launched (mode={mode}, morph={do_morph}, backend={'GPU' if _GPU else 'CPU'})")


def get_latest_fused_grid() -> xp.ndarray | None:
    """Return latest occupancy grid (CPU numpy array if copy=True)."""
    with _lock:
        return None if _latest_grid is None else _latest_grid.copy()


# ─────────────────── Command‑line entry ───────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(description="GPU fusion viewer / saver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--view", action="store_true", help="show live grids")
    p.add_argument("--mode", choices=["fixed", "scaled", "depthcam"], default="fixed")
    p.add_argument("--depth", type=float, default=20.0)
    p.add_argument("--h_fov", type=float, default=120.0)
    p.add_argument("--ids", type=int, nargs="*")
    p.add_argument("--morph", action="store_true")
    p.add_argument("--morph_size", type=int, default=3)
    p.add_argument("--vehicle", help="override vehicle name (config.yaml)")
    p.add_argument("--cam", help="override segmentation camera name")
    args = p.parse_args()

    global VEH, _cfg
    if args.vehicle:
        VEH = args.vehicle
        _cfg.vehicle = args.vehicle
    if args.cam:
        _cfg.seg_cam = args.cam
        if _cfg.segmentation:
            _cfg.segmentation["cam"] = args.cam

    _fusion_loop(mode=args.mode, depth_m=args.depth, h_fov_deg=args.h_fov,
                 ids=set(args.ids or DEFAULT_IDS), view=args.view,
                 do_morph=args.morph, morph_size=args.morph_size)

if __name__ == "__main__":
    _cli()
