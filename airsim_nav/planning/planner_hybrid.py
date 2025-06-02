# -*- coding: utf-8 -*-
"""
Reactive Planner  v0.3  (28 May 2025)  –  **Çarpışmasız Dur‑Kalk**
────────────────────────────────────────────────────────────────
Gelişmiş fren mantığı:
    v_allowed = √(2·a_max·(mesafe - güvenlik))
Bu, seçilen maksimum negatif ivme ile drone’un **daima** güvenlik
mesafesi (safe_distance) içinde durabileceğini garanti eder.

FSM:  MOVE (ileri)  ↔  TURN (yön değiştirme)
"""
from __future__ import annotations

import math, time
from typing import Optional

try:
    import cupy as xp
except ModuleNotFoundError:
    import numpy as xp  # CPU debug

import airsim
from airsim_nav.mapping.fusion_gpu import get_latest_fused_grid, start_fusion_thread
from airsim_nav.config import Params

_WRAP = lambda a: (a + math.pi) % (2 * math.pi) - math.pi  # [-π, π]

# ───────────────────────── Planner ────────────────────────────────────────
class ReactivePlanner:
    """MOVE / TURN FSM + kinematik fren güvenliği"""

    def __init__(self, cfg: Params):
        self.cs   = getattr(cfg, "cell_size", 0.25)
        self.safe = getattr(cfg, "safe_distance", 2.5)   # m
        self.look = getattr(cfg, "lookahead", 12.0)       # m
        self.Vmax = getattr(cfg, "V_forward", 4.0)        # m/s
        self.W    = math.radians(getattr(cfg, "Omega_turn", 60.0))  # rad/s
        self.acc  = getattr(cfg, "acc_max", 4.0)          # m/s² (fren kabiliyeti)
        self.sectors = 72
        self.mode = "MOVE"
        self.target_yaw: Optional[float] = None
        self.yaw_tol = math.radians(3)

    # ---- Engel mesafe (ray-cast) ----
    def _ray_dist(self, grid: xp.ndarray, ang: float, max_d: float) -> float:
        if grid is None: return 0.0
        G = grid.shape[0]; c = G//2
        dx = math.cos(ang)*self.cs; dy = math.sin(ang)*self.cs
        steps = int(max_d/self.cs)
        x=y=0.0
        for i in range(steps):
            gx=int(round(x/self.cs+c)); gy=int(round(y/self.cs+c))
            if not(0<=gx<G and 0<=gy<G): break
            if grid[gy,gx]: return i*self.cs
            x+=dx; y+=dy
        return max_d

    # ---- En yakın güvenli yön ----
    def _choose_dir(self, grid: xp.ndarray, yaw: float) -> Optional[float]:
        step = 2*math.pi/self.sectors
        for k in range(self.sectors//2+1):
            for sign in (1,-1) if k else (1,):
                ang = _WRAP(yaw+sign*k*step)
                if self._ray_dist(grid, ang, self.safe*2) >= self.safe*2:
                    return ang
        return None

    # ---- FSM step ----
    def step(self, yaw: float, grid: Optional[xp.ndarray]=None):
        if grid is None:
            grid = get_latest_fused_grid(False)
        if grid is None:
            return 0.,0.,0.

        d_fwd = self._ray_dist(grid, yaw, self.look)

        if self.mode == "MOVE":
            # kinematik limit: v² ≤ 2·a·(d - safe)
            margin = max(0.0, d_fwd - self.safe)
            v_allow = math.sqrt(2*self.acc*margin) if margin>0 else 0.0
            v_cmd = min(self.Vmax, v_allow)
            if v_cmd < 0.1:  # dur & yön değiştir
                new_ang = self._choose_dir(grid, yaw)
                if new_ang is None:
                    return 0.,0.,0.
                self.target_yaw = new_ang
                self.mode = "TURN"
                return 0.,0.,0.
            return v_cmd, 0., 0.

        elif self.mode == "TURN":
            if self.target_yaw is None:
                self.mode="MOVE"; return 0.,0.,0.
            diff = _WRAP(self.target_yaw - yaw)
            if abs(diff) < self.yaw_tol:
                self.mode="MOVE"; self.target_yaw=None
                return 0.,0.,0.
            w = self.W if diff>0 else -self.W
            return 0.,0.,w

        return 0.,0.,0.

# ───────────────────────── Navigator ──────────────────────────────────────

def demo():
    cfg = Params.load()
    cli = airsim.MultirotorClient(); cli.confirmConnection()
    cli.enableApiControl(True); cli.armDisarm(True)
    cli.takeoffAsync().join()
    cli.moveToZAsync(-getattr(cfg,"takeoff_alt",3.0),1).join()

    start_fusion_thread(view=False, mode="fixed")
    planner = ReactivePlanner(cfg)
    HZ = getattr(cfg,"control_hz",20); dt=1./HZ

    while True:
        st = cli.getMultirotorState()
        yaw = airsim.to_eularian_angles(st.kinematics_estimated.orientation)[2]
        v_body, vy_body, w = planner.step(yaw)
        vx = v_body*math.cos(yaw) - vy_body*math.sin(yaw)
        vy = v_body*math.sin(yaw) + vy_body*math.cos(yaw)
        cli.moveByVelocityAsync(vx, vy, 0, dt,
                                drivetrain=airsim.DrivetrainType.ForwardOnly,
                                yaw_mode=airsim.YawMode(True, math.degrees(w)))
        time.sleep(dt)

if __name__ == "__main__":
    demo()
