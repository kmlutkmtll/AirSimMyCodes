#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Waypoint Runner – 360° Seg Grid   (v2.2 • 02 Jun 2025)
"""

from __future__ import annotations
import math, time, heapq
from typing import List, Tuple
import numpy as np
import airsim
from airsim.utils import to_eularian_angles

from airsim_nav.config import Params
from airsim_nav.perception.seg360_thread import (
    start_seg360_thread, get_seg360_grid)

P = Params.load()
DT, GRID, CELL = 1.0 / P.control_hz, P.grid_dim, P.cell_size
CX = CY = GRID // 2
W_C, W_H, W_F, W_V = P.w_c, P.w_h, P.w_f, P.w_v
K_ATT, K_REP = P.k_attr, P.k_rep

# ────── A* yardımcıları ──────
def _grid_clear(path, g):
    G = g.shape[0]
    for (x0,y0),(x1,y1) in zip(path, path[1:]):
        dx,dy = abs(x1-x0), abs(y1-y0); sx,sy = (1 if x0<x1 else -1),(1 if y0<y1 else -1)
        err = dx - dy
        while True:
            if not (0<=x0<G and 0<=y0<G) or g[y0,x0]: return False
            if (x0,y0) == (x1,y1): break
            e2 = 2*err
            if e2 > -dy: err -= dy; x0 += sx
            if e2 <  dx: err += dx; y0 += sy
    return True

def _astar(st, gl, g):
    h = lambda x,y: abs(x-gl[0]) + abs(y-gl[1])
    open_set = [(h(*st),0,st,None)]
    came, cost = {}, {st:0}
    while open_set:
        _,c,n,p = heapq.heappop(open_set)
        if n == gl:
            path = [n]
            while p: path.append(p); p = came[p]
            return path[::-1]
        came[n] = p
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny = n[0]+dx,n[1]+dy
            if not (0<=nx<GRID and 0<=ny<GRID) or g[ny,nx]: continue
            nc = c+1
            if nc < cost.get((nx,ny),1e9):
                cost[(nx,ny)] = nc
                heapq.heappush(open_set,(nc+h(nx,ny),nc,(nx,ny),n))
    return []

# ────── Planlayıcı ──────
class Planner:
    def __init__(self, cli, wps):
        self.cli,self.wps = cli,wps
        self.i = 0
        self.v_prev = np.zeros(2,np.float32)
        self.yaw = 0.0
        self.last_clr = 1.0
        self.aeb_on = False

    def _clearance(self, vel, g):
        v = np.linalg.norm(vel)+1e-6
        steps = int(P.brake_distance / (v*DT))
        v_dir = vel / v * CELL
        p = np.array([CX,CY],np.float32)
        for i in range(1, steps+1):
            x,y = map(int, np.round(p + v_dir*i))
            if not (0<=x<GRID and 0<=y<GRID): break
            if g[max(0,y-1):min(GRID,y+2), max(0,x-1):min(GRID,x+2)].any():
                return i/steps
        return 1.0

    def _dwa(self, g):
        best_v,best_s = np.zeros(2), -1e9
        F_att = np.array([0, K_ATT])
        ys,xs = np.where(g)
        if xs.size:
            rel = np.stack([(xs-CX)*CELL, (ys-CY)*CELL], 1)
            d2  = (rel**2).sum(1)
            mask = d2 < (P.safe_distance+1)**2
            rep  = (-K_REP*(rel[mask]/(d2[mask,None]+1e-3)).sum(0)) if mask.any() else 0
        else:
            rep = 0
        F_des = F_att + rep

        for ang in np.linspace(-math.pi/4, math.pi/4, 17):
            dir_b = np.array([math.sin(ang), math.cos(ang)])      # body (+X ileri) -> grid +Y
            for v in np.linspace(0.2, P.V_max, 6):
                vel_b = v * dir_b
                vel_w = np.array([ vel_b[0]*math.cos(self.yaw) - vel_b[1]*math.sin(self.yaw),
                                   vel_b[0]*math.sin(self.yaw) + vel_b[1]*math.cos(self.yaw) ])
                head = np.dot(vel_w, F_des) / (np.linalg.norm(vel_w)+1e-6)
                clr  = self._clearance(vel_w, g); self.last_clr = clr
                score = W_H*head + W_C*clr + W_V*(v/P.V_max) + W_F*np.linalg.norm(F_des)
                if score > best_s:
                    best_s,best_v = score, vel_b
        self.v_prev = 0.6*self.v_prev + 0.4*best_v
        return self.v_prev

    def step(self, g):
        st = self.cli.getMultirotorState(vehicle_name=P.vehicle)
        self.yaw = to_eularian_angles(st.kinematics_estimated.orientation)[2]
        if np.linalg.norm(self.wps[self.i][:2]) < 0.8:
            self.i += 1
            if self.i >= len(self.wps):
                return np.zeros(2)
        return self._dwa(g)

# ────── Safety ──────
def safety(cli, g):
    st = cli.getMultirotorState(vehicle_name=P.vehicle)
    if st.collision.has_collided:
        cli.moveByVelocityBodyFrameAsync(0,0,0,1, vehicle_name=P.vehicle); return True
    if g is None: return False
    dist = planner.last_clr
    if dist < 0.15: planner.aeb_on = True
    elif dist > 0.30: planner.aeb_on = False
    if planner.aeb_on:
        cli.moveByVelocityBodyFrameAsync(0,0,0,1, vehicle_name=P.vehicle); return True
    band = 1
    if g[CY+1:CY+int(P.safe_distance/CELL)+1, CX-band:CX+band+1].any():
        cli.moveByVelocityBodyFrameAsync(0,0,0,1, vehicle_name=P.vehicle); return True
    return False

# ────── Main ──────
def main():
    global planner
    cli = airsim.MultirotorClient(); cli.confirmConnection()
    cli.enableApiControl(True, P.vehicle); cli.armDisarm(True, P.vehicle)
    cli.takeoffAsync(5, vehicle_name=P.vehicle).join()
    cli.moveToZAsync(-P.takeoff_alt, 1, vehicle_name=P.vehicle).join()

    start_seg360_thread(cam_h=2.0)

    wps = [np.array([x, y, -P.takeoff_alt], np.float32) for x,y in
          ([15,0],[15,15],[0,25],[-15,15],[-15,0],[-15,-15],
           [0,-25],[15,-15],[10,10],[-10,10],[-10,-10],[10,-10])]
    planner = Planner(cli, wps)

    while True:
        g = get_seg360_grid()
        if g is None or g.ndim != 2:
            time.sleep(0.05); continue
        if safety(cli, g):
            time.sleep(0.2); continue
        v = planner.step(g)
        cli.moveByVelocityBodyFrameAsync(float(v[0]), float(v[1]), 0,
                                         DT, vehicle_name=P.vehicle)
        time.sleep(DT)

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt:
        airsim.MultirotorClient().landAsync(vehicle_name=P.vehicle).join()
