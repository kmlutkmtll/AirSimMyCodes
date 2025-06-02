#!/usr/bin/env python3
# tests/quick_overlap.py
# ------------------------------------------------------------
# Lidar–Segmentation hizasını sayısal olarak ölçer:
#   • mask (pixel)  -> 300×300 grid
#   • lidar         -> occupancy grid
#   • overlap %     +  rot90  + flip testleri
# ------------------------------------------------------------
import math, sys
import airsim

# ------------ GPU varsa CuPy, yoksa NumPy -------------------
try:
    import cupy as xp
    _GPU = True
except ModuleNotFoundError:
    import numpy as xp
    _GPU = False

# ------------ Proje içi modüller ----------------------------
from airsim_nav.config import Params
from airsim_nav.perception.segmentation_processor_gpu import fetch_mask
from airsim_nav.perception.lidar_processor_gpu       import occupancy_grid
from airsim_nav.mapping.fusion_gpu                import _project_mask, GRID
# ------------------------------------------------------------

cfg  = Params.load()
VEH  = cfg.vehicle
print(f"[test] Config vehicle = {VEH}")

cli = airsim.MultirotorClient(); cli.confirmConnection()
print("[test] AirSim bağlı\n")

# ------------ 1) Segmentation maskesi (pixel) ---------------
seg_ids = set(cfg.segmentation["target_ids"])
mask_px = fetch_mask(cli, None, None, seg_ids, params=cfg)   # bool H×W
print("[test] Mask shape:", mask_px.shape)

# ------------ 2) Drone yaw (rad) ----------------------------
yaw = airsim.to_eularian_angles(
        cli.getMultirotorState(vehicle_name=VEH)
           .kinematics_estimated.orientation)[2]

# ------------ 3) Maskeyi 300×300 grid'e projekte et ---------
seg_grid = _project_mask(
    mask_px,
    yaw       = yaw,
    mode      = "fixed",                   # XY projeksiyon
    depth_m   = cfg.fusion.get("depth_max", 20),
    depth_map = None,
    h_fov     = math.radians(90)           # seg cam FOV
).astype(bool)

print("seg_grid.sum =", int(seg_grid.sum()))

# ------------ 4) Lidar occupancy grid -----------------------
ld = cli.getLidarData(lidar_name=cfg.lidar_name, vehicle_name=VEH)
if len(ld.point_cloud) < 3:
    sys.exit("[test] Lidar bulutu boş – sensör açık mı?")

pts = xp.asarray(ld.point_cloud, xp.float32).reshape(-1, 3)
lidar_g = occupancy_grid(pts, cfg).astype(bool)
print("lidar_g.sum  =", int(lidar_g.sum()))

# ------------ 5) Overlap % ----------------------------------
overlap = int((lidar_g & seg_grid).sum())
total   = int(seg_grid.sum())
percent = 100.0 * overlap / (total or 1)
print(f"\n[test] Lidar–Seg ÇAKIŞMA: {percent:.1f} % "
      f"({overlap}/{total} hücre)\n")

# ------------ 6) ROT90 taraması -----------------------------
print("ROT90 taraması (0/90/180/270 °):")
for k in (0, 1, 2, 3):
    ov = float(((lidar_g) &
                xp.rot90(seg_grid, k)).sum()) / (seg_grid.sum() + 1e-9)
    print(f"  rot {k*90:3}°  → {ov*100:5.1f} %")

# ------------ 7) FLIP testleri ------------------------------
print("\nFLIP testleri:")
for name, arr in [("fliplr", xp.fliplr(seg_grid)),      # sağ–sol
                  ("flipud", xp.flipud(seg_grid)),      # yukarı–aşağı
                  ("lr+ud",  xp.flipud(xp.fliplr(seg_grid)))]:
    ov = float((lidar_g & arr).sum()) / (arr.sum() + 1e-9)
    print(f"  {name:6}  → {ov*100:5.1f} %")

# ------------ 8) Opsiyonel görselleştirme -------------------
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,3))
    for i,(ttl,img,cmap) in enumerate(
        [("Lidar", lidar_g, "Reds"),
         ("Seg",   seg_grid, "Greens"),
         ("Union", (lidar_g|seg_grid), "gray_r")]):
        plt.subplot(1,3,i+1)
        plt.imshow(img.get() if _GPU else img,
                   cmap=cmap, origin="lower", interpolation="nearest")
        plt.title(ttl); plt.xticks([]); plt.yticks([])
    plt.show()
except ImportError:
    pass
