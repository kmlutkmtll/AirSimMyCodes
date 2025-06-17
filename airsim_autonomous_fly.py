#!usr/bin/env python3
# airsim_gap_avoid_safe_v2.py – Gap + Collision-Free Exploration (Stabilized + Fusion GPU)

import airsim
import numpy as np
import math, time, sys
from airsim_nav.config import Params
from airsim_nav.mapping.fusion_gpu import start_fusion_thread, get_latest_fused_grid

# ============================ Config yükle ==============================
_cfg = Params.load()
_gap_cfg = _cfg.gap_nav or {}

# Ayarları oku
FWD_SPEED         = _gap_cfg.get("fwd_speed",        5.0)
TTC_SEC           = _gap_cfg.get("ttc_sec",          1.4)
MIN_BRAKE_DIST    = _gap_cfg.get("min_brake_dist",   6.0)
CLEAR_DIST        = _gap_cfg.get("clear_dist",       9.0)
BRAKE_BACK_SPEED  = _gap_cfg.get("brake_back_speed", 8.0)
BRAKE_BACK_TIME   = _gap_cfg.get("brake_back_time",  1.0)
STEP_TIME         = _gap_cfg.get("step_time",        0.25)
FOV_SCAN          = _gap_cfg.get("fov_scan_deg",     200)
BINS              = _gap_cfg.get("gap_bins",         80)
SAFETY_MARGIN     = _gap_cfg.get("safety_margin",    1.0)

ALT_HOVER         = -abs(_cfg.takeoff_alt or 7.0)
LIDAR_NAME        = _cfg.lidar_name or "Lidar1"
VEHICLE           = _cfg.vehicle or "Drone1"

# ======================= AirSim yardımcıları ============================
def connect(v=""):
    cli = airsim.MultirotorClient(); cli.confirmConnection()
    cli.enableApiControl(True, v); cli.armDisarm(True, v)
    cli.simGetCollisionInfo()  # Güvenlik için çarpışma kapalı
    return cli

def takeoff(c, alt, v=""):
    print(f"[INFO] Takeoff başlatılıyor: hedef yükseklik {alt}m")
    c.takeoffAsync(vehicle_name=v).join()
    time.sleep(1.0)
    c.moveToZAsync(-3, 1.5, vehicle_name=v).join()
    c.moveToZAsync(alt, 2.5, vehicle_name=v).join()

def land(c, v=""):
    c.hoverAsync(vehicle_name=v).join()
    c.landAsync(vehicle_name=v).join()
    c.armDisarm(False, v)
    c.enableApiControl(False, v)

# ========================== Füzyon destekli LiDAR ======================
def _grid_to_points(grid: np.ndarray, cell: float = 0.25) -> np.ndarray:
    if grid is None or not grid.any():
        return np.empty((0, 3), np.float32)
    G = grid.shape[0]; cx = cy = G // 2
    ys, xs = np.nonzero(grid.astype(bool))
    dx = (xs - cx) * cell
    dy = (ys - cy) * cell
    return np.stack((dx, dy, np.zeros_like(dx)), axis=1).astype(np.float32)

def lidar_np(d):
    return np.array(d.point_cloud, np.float32).reshape(-1, 3) if len(d.point_cloud) >= 3 else np.empty((0, 3), np.float32)

def get_nav_cloud(cli):
    grid = get_latest_fused_grid()
    if grid is not None:
        cell = (_cfg.lidar or {}).get("cell_size", 0.25)
        return _grid_to_points(grid.get(), cell)
    return lidar_np(cli.getLidarData(LIDAR_NAME, VEHICLE))

# ========================== Geometri & Gap ============================
def front_min_dist(pts, fov=60):
    if pts.size == 0: return math.inf
    x, y, _ = pts.T; mask = x > 0
    if not mask.any(): return math.inf
    ang = np.degrees(np.arctan2(y[mask], x[mask]))
    sel = np.abs(ang) <= fov / 2
    if not sel.any(): return math.inf
    dist = np.linalg.norm(pts[mask][sel, :2], axis=1)
    return dist.min()

def best_gap_heading(pts, clear=CLEAR_DIST, fov=FOV_SCAN, bins=BINS):
    half, bin_size = fov / 2, fov / bins
    safe = np.ones(bins, bool); avg = np.zeros(bins)
    if pts.size:
        x, y, _ = pts.T
        ang = np.degrees(np.arctan2(y, x)); dist = np.hypot(x, y)
        mask = np.abs(ang) <= half
        ang, dist = ang[mask], dist[mask]
        idx = ((ang + half) / bin_size).astype(int).clip(0, bins - 1)
        for i, d in zip(idx, dist):
            if d < clear: safe[i] = False
            avg[i] += d
    best_score, best_center, i = -1.0, None, 0
    while i < bins:
        if safe[i]:
            start, total, cnt = i, 0.0, 0
            while i < bins and safe[i]:
                total += avg[i]; cnt += 1; i += 1
            mean_d = total / max(cnt, 1); score = cnt * mean_d
            if score > best_score:
                best_score = score; best_center = (start + i - 1) / 2
        else:
            i += 1
    if best_center is None: return None
    return (best_center * bin_size) - half

# ============================ Manevralar ===============================
def aggressive_brake(c, v=""):
    c.hoverAsync(vehicle_name=v).join()
    c.moveByVelocityBodyFrameAsync(-BRAKE_BACK_SPEED, 0, 0, BRAKE_BACK_TIME,
        drivetrain=airsim.DrivetrainType.ForwardOnly,
        yaw_mode=airsim.YawMode(False, 0), vehicle_name=v).join()
    c.hoverAsync(vehicle_name=v).join()

def ensure_clear(c, min_dist, v=""):
    while True:
        pts = get_nav_cloud(c)
        d = front_min_dist(pts)
        if d >= min_dist:
            return
        new_yaw = best_gap_heading(pts)
        if new_yaw is None:
            c.rotateByYawRateAsync(180, 1.5, vehicle_name=v).join()
        else:
            st = c.getMultirotorState(vehicle_name=v)
            cur = airsim.to_eularian_angles(st.kinematics_estimated.orientation)[2]
            tgt = math.degrees(cur) + new_yaw
            c.rotateToYawAsync(tgt, margin=5.0, vehicle_name=v).join()
        aggressive_brake(c, v)

# ============================== Ana Döngü ==============================
def explore(c):
    time.sleep(1.0)
    while True:
        c.moveByVelocityBodyFrameAsync(FWD_SPEED, 0, 0, STEP_TIME,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0), vehicle_name=VEHICLE).join()

        pts = get_nav_cloud(c)
        d = front_min_dist(pts)
        brake_dist = max(MIN_BRAKE_DIST, FWD_SPEED * TTC_SEC)
        if d > brake_dist:
            continue
        print(f"[WARN] Engel {d:.2f} m! → kaçınma")
        aggressive_brake(c, VEHICLE)
        ensure_clear(c, CLEAR_DIST + SAFETY_MARGIN, VEHICLE)
        st = c.getMultirotorState(vehicle_name=VEHICLE)
        if abs(st.kinematics_estimated.position.z_val - ALT_HOVER) > 0.5:
            c.moveToZAsync(ALT_HOVER, 2.0, vehicle_name=VEHICLE).join()

# ============================== Main ==================================
if __name__ == "__main__":
    cli = connect(VEHICLE)
    start_fusion_thread(view=False, mode="fixed", depth=20.0)
    try:
        takeoff(cli, ALT_HOVER, VEHICLE)
        explore(cli)
    except KeyboardInterrupt:
        print("[INFO] Kullanıcı durdurdu → iniş")
    finally:
        land(cli, VEHICLE)