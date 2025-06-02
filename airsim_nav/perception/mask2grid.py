import numpy as np

def mask_to_grid(mask: np.ndarray,
                 cam_h: float,
                 w: int, h: int, fov_deg: float,
                 grid_dim: int, cell: float) -> np.ndarray:
    """Seg maske → yere izdüşüm occupancy grid (1 = engel)."""
    fx = fy = 0.5 * w / np.tan(np.radians(fov_deg / 2))
    cx, cy = w / 2.0, h / 2.0

    ys, xs = np.where(mask > 200)            # beyaz pikseller
    x_cam = (xs - cx) / fx
    y_cam = (ys - cy) / fy                   # ekran y↓ → world +Y
    t = -cam_h                               # z_cam = +1
    Xw = x_cam * t                           # +X sağ
    Yw = y_cam * t                           # +Y ileri

    grid = np.zeros((grid_dim, grid_dim), np.uint8)
    gx = np.clip(np.round(Xw / cell).astype(int) + grid_dim // 2, 0, grid_dim-1)
    gy = np.clip(np.round(Yw / cell).astype(int) + grid_dim // 2, 0, grid_dim-1)
    grid[gy, gx] = 1
    return grid
