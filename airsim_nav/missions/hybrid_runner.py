#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Waypoint Autopilot – Continuous A* Re-planner (rev-03 Jun 2025, v0.1)
──────────────────────────────────────────────────────────────────────
• DWA yok – her 0.3 s’de bir grid üstünde A* yol, ilk 2–3 hücreyi takip
• Çarpışma önleme  : SafetyLayer  (+ clearance skoru)
• Fused grid GPU   : mapping.fusion_gpu (CuPy hızlandırmalı)
"""
from __future__ import annotations
import math, time, heapq, signal
from typing import List, Tuple
from threading import Event

import numpy as np
try:
    import cupy as cp; _HAS_CUPY = True
except ModuleNotFoundError:
    import numpy as cp; _HAS_CUPY = False

import airsim
from airsim.utils import to_eularian_angles

from airsim_nav.config import Params
from airsim_nav.mapping.fusion_gpu import (
    start_fusion_thread, get_latest_fused_grid)

# ─────────────── yardımcı dönüşüm ───────────────
def _rot_world_to_body(vec_w: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([vec_w[0]*c - vec_w[1]*s,
                     vec_w[0]*s + vec_w[1]*c], np.float32)

# ─────────────── parametreler ───────────────
P            = Params.load()
DT           = 1.0 / P.control_hz
GRID, CELL   = P.grid_dim, P.cell_size
CX = CY      = GRID // 2
BRAKE_A_MAX  = getattr(P, "a_max", 2.5)

# NumPy/CuPy yardımcıları
_xp = lambda a: cp.get_array_module(a) if (_HAS_CUPY and isinstance(a, cp.ndarray)) else np
_np = lambda a: a.get() if (_HAS_CUPY and isinstance(a, cp.ndarray)) else np.asarray(a)

# ─────────────── grid & A* ───────────────
def _grid_clear(path, grid):
    for (x0,y0),(x1,y1) in zip(path, path[1:]):
        dx,dy = abs(x1-x0), abs(y1-y0); sx,sy = (1 if x0<x1 else -1),(1 if y0<y1 else -1)
        err = dx - dy
        while True:
            if not (0<=x0<GRID and 0<=y0<GRID) or grid[y0,x0]:
                return False
            if (x0,y0)==(x1,y1): break
            e2 = 2*err
            if e2>-dy: err-=dy; x0+=sx
            if e2< dx: err+=dx; y0+=sy
    return True

def _astar(start: Tuple[int,int], goal: Tuple[int,int], grid: np.ndarray):
    grid = grid.copy()  # <-- ekle
    grid[start[1], start[0]] = 0  # <-- ekle
    grid[goal[1], goal[0]] = 0  # <-- ekle

    if grid[goal[1], goal[0]]: return []          # hedef doluysa yok
    h = lambda x,y: abs(x-goal[0])+abs(y-goal[1])
    open_set=[(h(*start),0,start,None)]; came, g_cost={},{start:0}
    while open_set:
        _,g,(x,y),prev = heapq.heappop(open_set)
        if (x,y)==goal:
            path=[(x,y)]
            while prev:
                path.append(prev)
                prev=came[prev]
            return path[::-1]
        came[(x,y)] = prev
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny = x+dx, y+dy
            if not (0<=nx<GRID and 0<=ny<GRID) or grid[ny,nx]: continue
            ng=g+1
            if ng < g_cost.get((nx,ny),1e9):
                g_cost[(nx,ny)]=ng
                heapq.heappush(open_set,(ng+h(nx,ny),ng,(nx,ny),(x,y)))
    return []

# ─────────────── planlayıcı ───────────────
class AStarPlanner:
    def __init__(self, cli, waypoints):
        self.cli = cli
        self.wps = waypoints
        self.idx = 0
        self.path: List[Tuple[int,int]] = []
        self.last_plan = 0.0
        self.v_prev = np.zeros(2, np.float32)
        self.yaw = 0.0
        self.last_clr = 1.0

    def _clearance(self, vel_w: np.ndarray, grid: np.ndarray) -> float:
        v = np.linalg.norm(vel_w)+1e-6
        steps = int(max(P.brake_distance, v**2/(2*BRAKE_A_MAX))/CELL)
        dir_w = vel_w/v
        p = np.array([CX,CY],np.float32)
        for i in range(1,steps+1):
            x,y = (p+dir_w*i).round().astype(int)
            if not (0<=x<GRID and 0<=y<GRID): break
            band=2
            if grid[max(0,y-band):min(GRID,y+band+1), max(0,x-band):min(GRID,x+band+1)].any():
                return i/steps
        return 1.0

    def step(self, grid_xp):
        if grid_xp is None or self.idx>=len(self.wps): return np.zeros(2,np.float32)
        grid = _np(grid_xp)

        st = self.cli.getMultirotorState(vehicle_name=P.vehicle)
        pos_w = np.array([st.kinematics_estimated.position.x_val,
                          st.kinematics_estimated.position.y_val],np.float32)
        self.yaw = to_eularian_angles(st.kinematics_estimated.orientation)[2]

        # WP tamamlandı mı?
        tgt = self.wps[self.idx]
        if np.linalg.norm(tgt[:2]-pos_w) < 0.8:
            self.idx += 1
            if self.idx>=len(self.wps): return np.zeros(2,np.float32)
            tgt = self.wps[self.idx]

        # Yol planla (0.3 s’de bir veya yol boşsa)
        if time.time()-self.last_plan > 0.3 or not self.path:
            goal_pix = ( (tgt[:2]-pos_w)/CELL + [CX,CY] ).astype(int)
            self.path = _astar((CX,CY), tuple(goal_pix), grid)
            self.last_plan = time.time()

        # Yol yoksa loiter
        if not self.path: return np.zeros(2,np.float32)

        # 2–3 hücre ötesine bak
        next_pix = np.array(self.path[min(2,len(self.path)-1)],np.float32)
        vec_pix  = next_pix - [CX,CY]
        vec_w    = vec_pix * CELL
        vec_b    = _rot_world_to_body(vec_w, self.yaw)
        if np.linalg.norm(vec_b) == 0: return np.zeros(2,np.float32)

        vel_b = vec_b / np.linalg.norm(vec_b) * P.V_max
        self.last_clr = self._clearance(vec_w, grid)
        self.v_prev = 0.6*self.v_prev + 0.4*vel_b   # filtre
        return self.v_prev

# ─────────────── safety layer ───────────────
def safety_brake(cli, grid_np, planner):
    if grid_np is None:
        return False
    st = cli.getMultirotorState(vehicle_name=P.vehicle)
    if st.collision.has_collided:
        cli.moveByVelocityBodyFrameAsync(0,0,0,1, vehicle_name=P.vehicle)
        print("[safety] collision flag!")
        return True
    if planner.last_clr < 0.3:
        cli.moveByVelocityBodyFrameAsync(0,0,0,1, vehicle_name=P.vehicle)
        print(f"[AEB] clearance {planner.last_clr:.2f}")
        return True
    band=2; cells=int(P.safe_distance/CELL)
    if grid_np[CY-band:CY+band+1, CX+1:CX+cells+1].any():
        cli.moveByVelocityBodyFrameAsync(0,0,0,1, vehicle_name=P.vehicle)
        print("[safety] Band Brake")
        return True
    return False

# ─────────────── main ───────────────
def main():
    cli = airsim.MultirotorClient(); cli.confirmConnection()
    cli.enableApiControl(True,P.vehicle); cli.armDisarm(True,P.vehicle)
    cli.takeoffAsync(timeout_sec=5, vehicle_name=P.vehicle).join()
    cli.moveToZAsync(-P.takeoff_alt,1,vehicle_name=P.vehicle).join()

    stop_evt: Event = start_fusion_thread(view=False, mode="fixed",
                                          do_morph=True, morph_size=3)

    waypoints = [[15,0,-P.takeoff_alt],[15,15,-P.takeoff_alt],
                 [0,25,-P.takeoff_alt],[-15,15,-P.takeoff_alt],
                 [-15,0,-P.takeoff_alt],[-15,-15,-P.takeoff_alt],
                 [0,-25,-P.takeoff_alt],[15,-15,-P.takeoff_alt]]
    planner = AStarPlanner(cli,[np.array(p,np.float32) for p in waypoints])

    def _sig(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _sig)

    try:
        while True:
            grid_xp = get_latest_fused_grid()
            grid_np = _np(grid_xp) if grid_xp is not None else None
            if safety_brake(cli, grid_np, planner):
                time.sleep(0.2); continue
            v_xy = planner.step(grid_xp)
            cli.moveByVelocityBodyFrameAsync(float(v_xy[0]), float(v_xy[1]), 0,
                                             DT, vehicle_name=P.vehicle)
            time.sleep(DT)
    except KeyboardInterrupt:
        print("\nLanding…")
    finally:
        if stop_evt: stop_evt.set()
        cli.moveByVelocityBodyFrameAsync(0,0,0,1, vehicle_name=P.vehicle).join()
        cli.landAsync(vehicle_name=P.vehicle).join()
        cli.enableApiControl(False,P.vehicle)
        print("Shutdown complete")

if __name__ == "__main__":
    main()
