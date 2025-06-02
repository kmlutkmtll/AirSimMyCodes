import airsim, numpy as np, cv2
from threading import Thread, Lock
from airsim_nav.config import Params
from .mask2grid import mask_to_grid

_cfg = Params.load()
cams = _cfg.segmentation.get("cams", ["front_center"])
intr = _cfg.segmentation["intrinsics"]              # w, h, fov_deg
orient = _cfg.segmentation["orient_map"]
GRID, CELL = _cfg.grid_dim, _cfg.cell_size

_lock = Lock()
_latest = {}                                        # cam → grid

def _worker(cam: str, cam_h: float):
    cli = airsim.MultirotorClient(); cli.confirmConnection()
    rot = orient[cam]["rotate"]; fh = orient[cam]["flip_h"]; fv = orient[cam]["flip_v"]
    while True:
        raw = cli.simGetImage(cam, airsim.ImageType.Segmentation)
        if raw is None: continue
        mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if rot: mask = np.rot90(mask, k=rot // 90)
        if fh:  mask = cv2.flip(mask, 1)
        if fv:  mask = cv2.flip(mask, 0)
        grid = mask_to_grid(mask, cam_h,
                            intr["w"], intr["h"], intr["fov_deg"],
                            GRID, CELL)
        with _lock:
            _latest[cam] = grid          # yalnız son kare tutulur

def _merge():
    g = None
    for v in _latest.values():
        g = v if g is None else np.logical_or(g, v)
    return None if g is None else g.astype(np.uint8)

def start_seg360_thread(cam_h: float = 2.0):
    for cam in cams:
        Thread(target=_worker, daemon=True, args=(cam, cam_h)).start()
    print(f"[seg360] started: {cams}")

def get_seg360_grid():
    with _lock:
        return _merge()
