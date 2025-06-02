#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lidar Processor – Density-Aware + Cluster-Adaptive Occupancy Grid
rev-27 May 2025, v0.7.1-cuda-opt
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional

try:
    import cupy as xp
    import cupyx.scipy.ndimage as xndi
    from cupy import RawKernel
    _GPU = True
except Exception:
    import numpy as xp
    import scipy.ndimage as xndi
    RawKernel = None
    _GPU = False

from airsim_nav.config import Params

@dataclass(slots=True)
class HistoParams:
    n_bins: int = 72
    max_range: float = 20.0
_H = HistoParams()

def _load_cfg(p: Optional[Params]) -> Tuple[float,int,float,int,float,int,int,float,bool,int]:
    if p is None:
        p = Params.load()
    c = getattr(p, "lidar", {})
    cs   = c.get("cell_size", 0.25)
    gd   = c.get("grid_dim", 300)
    zmin = c.get("z_min", -0.1)
    k    = c.get("density_k", 3)
    q    = c.get("quantile", 0.15)
    maxc = c.get("max_cost", 20)
    rit  = c.get("ransac_iter", 0)
    reps = c.get("ransac_thresh", 0.03)
    ray  = c.get("ray_free", True)
    blur = int(c.get("blur", 3))
    if blur <= 0:  # 0 (veya negatif) → filtreyi kapat
        blur = 0
    elif blur % 2 == 0:  # 2,4,6… gibi çiftse bir üst tek sayıya yuvarla
        blur += 1
    return cs, gd, zmin, k, q, maxc, rit, reps, ray, blur

if _GPU:
    _ray_kernel = RawKernel(r'''
    extern "C" __global__
    void raycast(const int cx, const int cy,
                 const int* gx, const int* gy,
                 unsigned char* frees, const int size, const int gd)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= size) return;

        int r0 = cy, c0 = cx;
        int r1 = gy[i], c1 = gx[i];
        int dr = abs(r1 - r0), dc = abs(c1 - c0);
        int sr = (r0 < r1) ? 1 : -1;
        int sc = (c0 < c1) ? 1 : -1;
        int err = dr - dc;

        while (true) {
            if (r0 == r1 && c0 == c1) break;
            if (r0 >= 0 && r0 < gd && c0 >= 0 && c0 < gd)
                frees[r0 * gd + c0] = 1;

            int e2 = 2 * err;
            if (e2 > -dc) { err -= dc; r0 += sr; }
            if (e2 <  dr) { err += dr; c0 += sc; }
        }
    }
    ''', 'raycast')
else:
    _ray_kernel = None

def _remove_ground(pts: xp.ndarray, it: int, eps: float) -> xp.ndarray:
    if pts.shape[0] < 50 or it == 0:
        return pts
    best = xp.zeros(pts.shape[0], bool)
    best_cnt = 0
    for _ in range(it):
        idx = xp.random.randint(0, pts.shape[0], size=3)
        p1, p2, p3 = pts[idx]
        n = xp.cross(p2 - p1, p3 - p1)
        norm = xp.linalg.norm(n)
        if norm < 1e-6:
            continue
        n /= norm
        d = -xp.dot(n, p1)
        dist = xp.abs(pts @ n + d)
        inl = dist < eps
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt, best = cnt, inl
            if cnt > pts.shape[0] * 0.8:
                break
    # Eğer zemin inlier sayısı çok azsa (ör. < %2) zemini yok say,
    # tüm noktaları geçerli bırak.  Böylece "her yer engel" patlaması
    # yaşanmaz, sadece zemin filtresi devre dışı kalır.
    if best_cnt < int(pts.shape[0] * 0.02):
        return pts      # zemin çıkarma başarısız → ham bulut

    return pts[~best]   # normal durumda zemini at


def occupancy_cost(pts, p: Params | None = None) -> xp.ndarray:
    pts = xp.asarray(pts)
    cs, gd, zmin, _, _, maxc, rit, reps, _, _ = _load_cfg(p)
    if pts.size == 0:
        return xp.zeros((gd, gd), xp.uint8)
    pts = _remove_ground(pts, rit, reps)
    pts = pts[pts[:, 2] > zmin]
    if pts.size == 0:
        return xp.zeros((gd, gd), xp.uint8)

    cx = cy = gd // 2
    gx = xp.floor(pts[:, 0] / cs + cx).astype(xp.int32)
    gy = xp.floor(pts[:, 1] / cs + cy).astype(xp.int32)
    mask = (gx >= 0) & (gx < gd) & (gy >= 0) & (gy < gd)
    idx = gy[mask] * gd + gx[mask]
    counts = xp.bincount(idx, minlength=gd * gd).reshape(gd, gd)
    return xp.clip(counts, 0, maxc).astype(xp.uint8)

def occupancy_grid(pts, p: Params | None = None) -> xp.ndarray:
    pts = xp.asarray(pts)
    cs, gd, zmin, kfix, q, _, rit, reps, ray_free, blur = _load_cfg(p)

    cost = occupancy_cost(pts, p)
    smooth = (xndi.convolve(cost.astype(xp.float32),
                            xp.ones((blur, blur), xp.float32),
                            mode="constant") if blur > 1 else cost.astype(xp.float32))
    nz = smooth[smooth > 0]
    thr = max(kfix, float(xp.quantile(nz, 1 - q)) if (q > 0 and nz.size) else 0)
    occ = (smooth >= thr).astype(xp.uint8)

    if not ray_free or pts.size == 0:
        return occ

    cx = cy = gd // 2
    gx = xp.floor(pts[:, 0] / cs + cx).astype(xp.int32)
    gy = xp.floor(pts[:, 1] / cs + cy).astype(xp.int32)
    mask = (gx >= 0) & (gx < gd) & (gy >= 0) & (gy < gd)
    gx, gy = gx[mask], gy[mask]
    if gx.size == 0:
        return occ
    frees = xp.zeros((gd, gd), dtype=xp.uint8)

    if _GPU:
        d_gx = gx.astype(xp.int32)
        d_gy = gy.astype(xp.int32)
        d_frees = xp.zeros(gd * gd, dtype=xp.uint8)
        threads = 256
        blocks = (gx.size + threads - 1) // threads
        _ray_kernel((blocks,), (threads,),
                    (int(cx), int(cy), d_gx, d_gy, d_frees, gx.size, gd))
        frees = d_frees.reshape((gd, gd))
    else:
        for xi, yi in zip(gx.tolist(), gy.tolist()):
            r0, c0 = cy, cx
            r1, c1 = yi, xi
            dr, dc = abs(r1 - r0), abs(c1 - c0)
            sr, sc = (1 if r0 < r1 else -1), (1 if c0 < c1 else -1)
            err = dr - dc
            while True:
                if r0 == r1 and c0 == c1: break
                frees[r0, c0] = True
                e2 = err * 2
                if e2 > -dc: err -= dc; r0 += sr
                if e2 < dr: err += dr; c0 += sc

    occ[(frees) & (cost == 0)] = 0
    return occ

def polar_hist_grid(grid: xp.ndarray, h: HistoParams = _H) -> xp.ndarray:
    if not grid.any():
        return xp.zeros(h.n_bins, xp.int32)
    G = grid.shape[0]; cx = cy = G // 2
    ys, xs = xp.nonzero(grid.astype(bool))
    dx, dy = xs - cx, ys - cy
    dist = xp.sqrt(dx * dx + dy * dy)
    cs = _load_cfg(None)[0]
    mask = dist <= h.max_range / cs
    if not mask.any():
        return xp.zeros(h.n_bins, xp.int32)
    ang = xp.arctan2(dy[mask], dx[mask])
    ang[ang < 0] += 2 * math.pi
    bins = xp.floor(ang / (2 * math.pi / h.n_bins)).astype(xp.int32)
    return xp.bincount(bins, minlength=h.n_bins).astype(xp.int32)

if __name__ == "__main__":
    import numpy as np
    print("GPU mode:", _GPU)
    ground = np.array([[x, y, 0] for x in np.linspace(-5, 5, 40)
                                  for y in np.linspace(-5, 5, 40)], np.float32)
    cube   = np.array([[x, y, z] for x in np.linspace(1, 3, 15)
                                  for y in np.linspace(-1, 1, 15)
                                  for z in np.linspace(0, 2, 8)], np.float32)
    pole   = np.array([[0, 4, z] for z in np.linspace(0, 3, 12)], np.float32)
    pts    = xp.asarray(np.concatenate([ground, cube, pole]))
    grid   = occupancy_grid(pts)
    if _GPU:
        xp.cuda.Stream.null.synchronize()
    print("occ %:", float(grid.mean() * 100))
    print("z min / max:", pts[:, 2].min(), pts[:, 2].max())
