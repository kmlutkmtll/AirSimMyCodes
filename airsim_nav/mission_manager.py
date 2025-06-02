#!/usr/bin/env python3
"""
Mission Waypoint Runner – v0.6 (Hybrid APF⇆DWA, Camera‑Aligned, Safe Brake)
────────────────────────────────────────────────────────────────────────────
• Kamera/burun yönü artık hız vektörüyle tam eşleşir (YAW_OFFSET).
• Daha erken kaçınma (`CLEAR_THRESH = 4 m`), güvenli fren (`BRAKE_THRESH = 1.2 m`).
• `dwa.clear_gain = 3.0` → engel mesafesine daha duyarlı.
• Tek dosya, tam kapalı; IDE’de sentaks hatasız.
"""
from __future__ import annotations

import argparse, math, time, heapq, yaml
from pathlib import Path
from typing import List, Tuple

import numpy as np
import airsim
try:
    import cupy as cp
    import cupyx.scipy.ndimage as xndi  # GPU backend
except ModuleNotFoundError:
    cp = None
    import scipy.ndimage as xndi        # CPU fallback

from airsim_nav.config import Params
from airsim_nav.mapping.fusion_gpu import start_fusion_thread, get_latest_fused_grid
from airsim_nav.perception.lidar_processor_gpu import polar_hist_grid

###############################################################################
# Konfigürasyon & Sabitler
###############################################################################

CFG = Params.load()
CELL_SZ = CFG.cell_size            # Occupancy grid hücre boyu (metre)
YAW_OFFSET = -90.0                 # ° Drone mesh’i +Y yönüne bakıyorsa -90

DWA_DEF = dict(dt=0.2, v_samples=7, omega_samples=9,
               max_acc=2.0,
               max_ang_acc=math.radians(90),
               heading_gain=3.0, vel_gain=1.0, clear_gain=3.0)

CLEAR_THRESH = 4.0   # m  → bu mesafeden daha yakında DWA kullan
BRAKE_THRESH  = 1.2  # m  → bu mesafeden daha yakında hız vektörü sıfır

###############################################################################
# Yardımcı Fonksiyonlar
###############################################################################

def ned_dist_xy(p1, p2):
    return math.hypot(p1.x_val - p2.x_val, p1.y_val - p2.y_val)

def reached(pos, wp, tol):
    return ned_dist_xy(pos, airsim.Vector3r(wp['x'], wp['y'], wp['z'])) <= tol['xy'] \
        and abs(pos.z_val - wp['z']) <= tol['z']

def ned_to_cell(dx, dy, cx, cy):
    return cx - int(dx / CELL_SZ), cy + int(dy / CELL_SZ)

def cell_to_ned(r, c, cx, cy):
    return (cx - r + 0.5) * CELL_SZ, (c - cy + 0.5) * CELL_SZ

def _xp(arr):
    return cp if cp and isinstance(arr, cp.ndarray) else np

###############################################################################
# Global Planlayıcı (A*)
###############################################################################

def a_star(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    h, w = grid.shape
    sr, sc = start; gr, gc = goal
    if not (0 <= gr < h and 0 <= gc < w) or grid[gr, gc]:
        return []

    open_set = [(abs(gr - sr) + abs(gc - sc), 0.0, sr, sc, None)]
    g_score = {(sr, sc): 0.0}
    came: dict[Tuple[int, int], Tuple[int, int] | None] = {}
    closed = set()

    while open_set:
        f, g, r, c, par = heapq.heappop(open_set)
        if (r, c) in closed:
            continue
        came[(r, c)] = par
        if (r, c) == (gr, gc):
            break
        closed.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or grid[nr, nc]:
                continue
            ng = g + math.hypot(dr, dc)
            if ng < g_score.get((nr, nc), 1e9):
                g_score[(nr, nc)] = ng
                f = ng + math.hypot(gr - nr, gc - nc)
                heapq.heappush(open_set, (f, ng, nr, nc, (r, c)))

    path: List[Tuple[int, int]] = []
    cur: Tuple[int, int] | None = (gr, gc)
    while cur and cur in came:
        path.append(cur)
        cur = came[cur]
    path.reverse()
    return path

def path_blocked(path, grid):
    return any(grid[r, c] for r, c in path)

###############################################################################
# Lokal Planlayıcılar
###############################################################################

# — APF

def local_apf(goal_ned, grid):
    xp = _xp(grid)
    att = xp.asarray(goal_ned, dtype=xp.float32)
    att /= xp.linalg.norm(att) + 1e-6
    hist = polar_hist_grid(grid)
    rep = xp.zeros(2, dtype=xp.float32)
    max_cnt = float(hist.max()) or 1.0
    for i, cnt in enumerate(hist):
        if cnt == 0:
            continue
        ang = 2 * math.pi * i / len(hist)
        rep -= (cnt / max_cnt) * xp.asarray([math.cos(ang), math.sin(ang)], xp.float32)
    v = CFG.k_attr * att + CFG.k_rep * rep
    speed = float(xp.linalg.norm(v))
    if speed > CFG.max_vel_fwd:
        v *= CFG.max_vel_fwd / speed
    if xp is cp:
        v = cp.asnumpy(v)
    return float(v[0]), float(v[1])

# — DWA

def _dist_field(bin_grid):
    free = (bin_grid == 0).astype(bin_grid.dtype)
    return xndi.distance_transform_edt(free) * CELL_SZ

def _heading(g, tr):
    return (1 + np.dot(g, tr) / (np.linalg.norm(g) * np.linalg.norm(tr) + 1e-6)) / 2

def local_dwa(state, goal_xy, bin_g, cfg):
    vx0, vy0 = state.linear_velocity.x_val, state.linear_velocity.y_val
    w0 = state.angular_velocity.z_val
    dt = cfg['dt']
    V = np.linspace(vx0 - cfg['max_acc'] * dt, vx0 + cfg['max_acc'] * dt, cfg['v_samples'])
    Vy = np.linspace(vy0 - cfg['max_acc'] * dt, vy0 + cfg['max_acc'] * dt, cfg['v_samples'])
    Ws = np.linspace(w0 - cfg['max_ang_acc'] * dt, w0 + cfg['max_ang_acc'] * dt, cfg['omega_samples'])
    df = _dist_field(bin_g)
    h, w = df.shape
    best_score, best_cmd = -1e9, (0, 0, 0)
    gv = np.asarray(goal_xy)
    for vx in V:
        for vy in Vy:
            for om in Ws:
                x, y = vx * dt, vy * dt
                score = cfg['heading_gain'] * _heading(gv, (x, y)) + cfg['vel_gain'] * math.hypot(vx, vy)
                ix = int(h / 2 - y / CELL_SZ)
                iy = int(w / 2 + x / CELL_SZ)
                clear = df[ix, iy] if 0 <= ix < h and 0 <= iy < w else 0.0
                score += cfg['clear_gain'] * clear
                if score > best_score:
                    best_score, best_cmd = score, (vx, vy, om)
    return best_cmd

###############################################################################
# Yardımcı Actions
###############################################################################

def do_actions(cli, acts):
    if not acts:
        return
    for a in acts:
        if a == 'arm':
            cli.enableApiControl(True); cli.armDisarm(True)
        elif a == 'disarm':
            cli.armDisarm(False); cli.enableApiControl(False)
        elif a == 'land':
            cli.landAsync().join()

###############################################################################
# Ana Görev Koşucu
###############################################################################
def run_mission(file: str, vehicle: str, force_dwa: bool):
    mission = yaml.safe_load(Path(file).read_text(encoding="utf-8"))
    tol = mission["meta"]["reach_tol"]
    dwa_cfg = {**DWA_DEF, **getattr(CFG, "dwa", {})}

    cli = airsim.MultirotorClient()
    cli.confirmConnection()
    start_fusion_thread()

    dt = 1 / CFG.control_hz          # kontrol çevrimi
    repl_period = 0.4                # s — global re-plan
    VZ = 1.0                         # m/s – sabit dikey hız

    for wp in mission["waypoints"]:
        goal = wp["pos"]
        do_actions(cli, wp.get("action"))

        path: List[Tuple[int, int]] = []
        idx = 0
        last_plan_t = time.time()
        dist_f = None                # mesafe alanı (clearance)

        while True:
            state = cli.getMultirotorState().kinematics_estimated
            if reached(state.position, goal, tol):
                break

            grid = get_latest_fused_grid()
            if grid is None:
                time.sleep(0.05)
                continue
            bin_g = (grid > 0).astype(np.uint8)

            h, w = bin_g.shape
            cx, cy = h // 2, w // 2

            dx = goal["x"] - state.position.x_val
            dy = goal["y"] - state.position.y_val
            dz = goal["z"] - state.position.z_val

            # --- Küresel plan / yeniden-plan ---------------------------------
            if (not path) or (
                time.time() - last_plan_t > repl_period and path_blocked(path, bin_g)
            ):
                path = a_star(bin_g, (cx, cy), ned_to_cell(dx, dy, cx, cy))
                idx = 0
                last_plan_t = time.time()

            # --- Alt hedef seçimi -------------------------------------------
            if not path:
                sub_dx, sub_dy = dx, dy
            else:
                idx = min(idx, len(path) - 1)
                r, c = path[idx]
                sub_dx, sub_dy = cell_to_ned(r, c, cx, cy)
                if math.hypot(sub_dx, sub_dy) < 0.6 * CELL_SZ and idx < len(path) - 1:
                    idx += 1

            # --- Mesafe alanı & güvenlik ------------------------------------
            if dist_f is None or time.time() - last_plan_t > 1.0:
                dist_f = _dist_field(bin_g)
            clearance = dist_f[cx, cy]

            # --- Yerel planlayıcı seçimi ------------------------------------
            use_dwa = force_dwa or clearance < CLEAR_THRESH
            if use_dwa:
                vx, vy, omega = local_dwa(state, (sub_dx, sub_dy), bin_g, dwa_cfg)
            else:
                vx, vy = local_apf((sub_dx, sub_dy), grid)
                omega = 0.0

            # --- Acil fren ---------------------------------------------------
            if clearance < BRAKE_THRESH:
                vx = vy = omega = 0.0

            # --- Dikey kontrol ----------------------------------------------
            vz = 0.0
            if abs(dz) > tol["z"]:
                vz = -VZ if dz < 0 else VZ
                if abs(dz) < VZ * dt:
                    vz = dz / dt  # yumuşak yaklaşım

            # --- Yaw: hız yönüne bak ----------------------------------------
            if math.hypot(vx, vy) > 0.05:
                desired_yaw = math.degrees(math.atan2(vy, vx))
            else:
                desired_yaw = math.degrees(
                    airsim.to_eularian_angles(state.orientation)[2]
                )
            yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=desired_yaw)

            # --- Komut gönder ------------------------------------------------
            cli.moveByVelocityAsync(
                vx,
                vy,
                vz,
                duration=dt,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=yaw_mode,
                vehicle_name=vehicle,
            )
            time.sleep(dt)

        # waypoint'te bekleme süresi
        if wp.get("hold"):
            cli.hoverAsync(vehicle_name=vehicle).join()
            time.sleep(wp["hold"])

    # Görev bitti → iniş
    cli.landAsync(vehicle_name=vehicle).join()
    cli.armDisarm(False, vehicle_name=vehicle)
    cli.enableApiControl(False, vehicle_name=vehicle)
###############################################################################
# CLI
###############################################################################

if __name__=='__main__':
    p=argparse.ArgumentParser();
    p.add_argument('mission_yaml'); p.add_argument('--vehicle',default='Drone1');
    p.add_argument('--dwa',action='store_true',help='force DWA everywhere');
    a=p.parse_args(); run_mission(a.mission_yaml,a.vehicle,a.dwa)
