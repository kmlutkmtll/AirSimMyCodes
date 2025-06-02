#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lidar Processor – Density‑Aware + Cluster‑Adaptive Occupancy Grid (rev‑22 May 2025, v0.6)
========================================================================================
Bu sürüm v0.5 üzerine üç büyük iyileştirme getirir:

1. **Ray‑cast free‑space** (`ray_free: true`) korunur – ışın boyunca boş
   hücreler serbest bırakılır.
2. **Yoğunluk yumuşatma** (`blur: 3|5|7`) – ham vuruş haritası komşuluk
   kernel’i (3 × 3 / 5 × 5 / 7 × 7) ile filtrelenir; tek tük vuruşlar
   engel sayılmaz, gerçek kümeler öne çıkar.
3. **Çift eşik mantığı**
   • `density_k`   – smoothed haritada sabit minimum vuruş sayısı
   • `quantile`    – smoothed haritanın en yoğun %X’i engel

YAML Örneği
-----------
```yaml
lidar:
  cell_size:   0.5
  grid_dim:    300
  ray_free:    true   # ışın free‑space
  blur:        5      # 3|5|7 – büyük ⇒ daha selektif
  density_k:   5      # ≥5 (smoothed) vuruş ⇒ engel
  quantile:    0.25   # en yoğun %25 ⇒ engel (0 → kapalı)
```
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
import scipy.ndimage as ndi

from airsim_nav.config import Params

# ───────────────────── Histogram Ayarları ─────────────────────
@dataclass(slots=True)
class HistoParams:
    n_bins: int = 72           # 5° çözünürlük
    max_range: float = 20.0    # (m)

_default_hist = HistoParams()

# ───────────── Parametre Yükleyici ─────────────

def _load_cfg(params: Optional[Params]) -> Tuple[float,int,float,int,float,int,int,float,bool,int]:
    """Lidar konfigürasyonunu topla."""
    if params is None:
        params = Params.load()
    cfg = getattr(params, "lidar", None) or {}
    cs   = cfg.get("cell_size", 0.5)
    gd   = cfg.get("grid_dim", 300)
    zmin = cfg.get("z_min", -0.1)
    k    = cfg.get("density_k", 5)
    q    = cfg.get("quantile", 0.25)
    maxc = cfg.get("max_cost", 20)
    rit  = cfg.get("ransac_iter", 0)
    reps = cfg.get("ransac_thresh", 0.03)
    ray  = cfg.get("ray_free", True)
    blur = int(cfg.get("blur", 5)) if cfg.get("blur", 5) % 2 == 1 else 5
    return cs, gd, zmin, k, q, maxc, rit, reps, ray, blur

# ─────────── RANSAC Ground Filter ───────────

def _remove_ground_ransac(pts: np.ndarray, max_iter: int, eps: float) -> np.ndarray:
    N = pts.shape[0]
    if N < 50 or max_iter == 0:
        return pts
    rng = np.random.default_rng()
    best_inl = None; best_cnt = 0
    for _ in range(max_iter):
        idx = rng.choice(N, 3, replace=False)
        p1,p2,p3 = pts[idx]
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-6:
            continue
        n /= n_norm; d = -np.dot(n, p1)
        dist = np.abs(pts @ n + d)
        inl = dist < eps; cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt, best_inl = cnt, inl
            if best_cnt > N*0.8:
                break
    return pts if best_inl is None else pts[~best_inl]

# ─────────── Bresenham ───────────

def _bresenham(r0: int, c0: int, r1: int, c1: int):
    dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    rr, cc = [], []
    while True:
        rr.append(r0); cc.append(c0)
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc; r0 += sr
        if e2 < dr:
            err += dr; c0 += sc
    return np.array(rr), np.array(cc)

# ─────────── Occupancy Cost ───────────

def occupancy_cost(pts: np.ndarray, params: Params | None = None) -> np.ndarray:
    cs, gd, zmin, _, _, maxc, rit, reps, _, _ = _load_cfg(params)
    if pts.size == 0:
        return np.zeros((gd, gd), np.uint8)
    pts_ng = _remove_ground_ransac(pts, rit, reps)
    xy = pts_ng[pts_ng[:,2] > zmin, :2]
    if xy.size == 0:
        return np.zeros((gd, gd), np.uint8)
    cx = cy = gd // 2
    gx = np.floor(xy[:,0]/cs + cx).astype(np.int32)
    gy = np.floor(xy[:,1]/cs + cy).astype(np.int32)
    mask = (0<=gx)&(gx<gd)&(0<=gy)&(gy<gd)
    gx,gy = gx[mask], gy[mask]
    counts = np.bincount(gy*gd + gx, minlength=gd*gd).reshape(gd,gd)
    return np.clip(counts, 0, maxc).astype(np.uint8)

# ─────────── Occupancy Grid ───────────

def occupancy_grid(pts: np.ndarray, params: Params | None = None) -> np.ndarray:
    cs, gd, zmin, kfix, q, _, rit, reps, ray_free, blur = _load_cfg(params)
    cost = occupancy_cost(pts, params)
    # 1) Yoğunluk yumuşatma
    if blur > 1:
        kernel = np.ones((blur, blur), np.float32)
        smooth = ndi.convolve(cost.astype(np.float32), kernel, mode="constant")
    else:
        smooth = cost.astype(np.float32)
    # 2) Eşikleme
    thr_k = kfix
    thr_q = 0
    if q > 0 and np.any(smooth>0):
        thr_q = np.quantile(smooth[smooth>0], 1 - q)
    thr = max(thr_k, thr_q)
    occ = (smooth >= thr).astype(np.uint8)
    if not ray_free or pts.size == 0:
        return occ
    # 3) Ray‑cast free-space
    cx = cy = gd // 2
    gx_all = np.floor(pts[:,0]/cs + cx).astype(np.int32)
    gy_all = np.floor(pts[:,1]/cs + cy).astype(np.int32)
    mask = (0<=gx_all)&(gx_all<gd)&(0<=gy_all)&(gy_all<gd)
    gx_all, gy_all = gx_all[mask], gy_all[mask]
    frees = np.zeros_like(occ, np.bool_)
    for gx,gy in zip(gx_all, gy_all):
        rr,cc = _bresenham(cy,cx,gy,gx)
        # son hücre (engel) hariç free
        frees[rr[:-1], cc[:-1]] = True
    occ[frees] = 0
    return occ

# ─────────── Polar Histogram ───────────

def polar_hist_grid(grid: np.ndarray, hist_cfg: HistoParams = _default_hist) -> np.ndarray:
    g = grid.astype(bool)
    if not g.any():
        return np.zeros(hist_cfg.n_bins, np.int32)
    G = grid.shape[0]; cx = cy = G//2
    ys,xs = np.nonzero(g)
    dx,dy = xs-cx, ys-cy
    dist  = np.hypot(dx,dy)
    cs = _load_cfg(None)[0]
    max_cells = hist_cfg.max_range / cs
    mask = dist <= max_cells
    if not np.any(mask):
        return np.zeros(hist_cfg.n_bins, np.int32)
    ang = np.arctan2(dy[mask], dx[mask])
    ang[ang<0] += 2*math.pi
    bins = np.floor(ang / (2*math.pi/hist_cfg.n_bins)).astype(np.int32)
    return np.bincount(bins, minlength=hist_cfg.n_bins).astype(np.int32)

# ─────────── Hızlı Test ───────────
if __name__ == "__main__":
    ground = np.array([[x,y,0] for x in np.linspace(-5,5,40) for y in np.linspace(-5,5,40)], np.float32)
    cube   = np.array([[x,y,z] for x in np.linspace(1,3,15) for y in np.linspace(-1,1,15) for z in np.linspace(0,2,8)], np.float32)
    pole   = np.array([[0,4,z] for z in np.linspace(0,3,12)], np.float32)
    pts = np.vstack([ground, cube, pole])
    g = occupancy_grid(pts)
    print("occ %:", g.mean()*100)
