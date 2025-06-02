#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geometry.py  –  Basit vektörel yardımcılar
──────────────────────────────────────────
Reaktif + proaktif planlama aşamalarında ortak kullanılacak:
    • açı sarmalama  (wrap_pi / deg↔rad)
    • 2-B nokta dönüşü (gövde⇄dünya)
    • grid üzerinde AABB çarpışma testi
"""

from __future__ import annotations
import math
from typing import Tuple

import numpy as np

# ── Açı yardımcıları ───────────────────────────────────────────────
def wrap_pi(angle: float) -> float:
    """Açıyı (rad) −π … +π aralığına sarar."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

deg2rad = lambda d: d * math.pi / 180.0
rad2deg = lambda r: r * 180.0 / math.pi

# ── Koordinat dönüşümü (2-B) ───────────────────────────────────────
def body_to_world(pt_body: Tuple[float, float], yaw: float) -> Tuple[float, float]:
    """(x_fwd, y_right) → dünya çerçevesine döndür."""
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = pt_body
    return (c * x - s * y, s * x + c * y)

def world_to_body(pt_world: Tuple[float, float], yaw: float) -> Tuple[float, float]:
    """Dünya → gövde dönüşümü."""
    c, s = math.cos(-yaw), math.sin(-yaw)
    x, y = pt_world
    return (c * x - s * y, s * x + c * y)

# ── Grid tabanlı engel testleri ───────────────────────────────────
def rect_collision_free(grid: np.ndarray,
                        cx: int, cy: int,
                        rad: int) -> bool:
    """
    (cx,cy) merkezli kare kabukta ‘1’ var mı?
    rad : yarıçap hücre (>=0)
    True → güvenli, False → çarpışma.
    """
    y0 = max(cy - rad, 0); y1 = min(cy + rad + 1, grid.shape[0])
    x0 = max(cx - rad, 0); x1 = min(cx + rad + 1, grid.shape[1])
    return not grid[y0:y1, x0:x1].any()

def inflate_obstacles(grid: np.ndarray, rad: int) -> np.ndarray:
    """
    Engelleri rad hücre genişlet (binary dilation, CPU-hafif).
    """
    if rad <= 0:
        return grid.copy()
    pad = np.pad(grid, rad, mode="constant", constant_values=0)
    kernel = np.ones((2*rad+1, 2*rad+1), dtype=np.uint8)
    from numpy.lib.stride_tricks import sliding_window_view as swv
    view = swv(pad, kernel.shape)
    return (view * kernel).max(axis=(-1, -2)).astype(np.uint8)

__all__ = [
    "wrap_pi", "deg2rad", "rad2deg",
    "body_to_world", "world_to_body",
    "rect_collision_free", "inflate_obstacles"
]
