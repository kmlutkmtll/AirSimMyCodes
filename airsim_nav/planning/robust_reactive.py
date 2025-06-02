#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geliştirilmiş Hybrid Navigator + Path Following
----------------------------------------------
1) Fusion thread (Lidar+Seg -> occupancy, morph filtreli).
2) Collision varsa -> emergency_escape.
3) Danger/Stuck -> Plan (largest_free_space) -> fallback multi-goals.
4) Bulunan yol (path) waypoint'leri takip eder, her döngüde
   "hemen planlama" yerine "waypoint'e yaklaşma" mantığı çalışır.
5) Plan bozulursa (collision, tehlike) -> gap-based / escalate devreye girer.
"""

import argparse
import math
import time
import random
from collections import deque
import heapq

import numpy as np
import airsim

# Fusion modülü (fusion.py) - parametre uyumlu olmalı!
from airsim_nav.mapping.fusion_gpu import (
    start_fusion_thread,
    get_latest_fused_grid,
)

# -----------------------------------------------------------------------------
# Parametreler
# -----------------------------------------------------------------------------
VEHICLE         = "Drone1"
TAKEOFF_ALT     = 7
CRUISE_VEL_FWD  = 4.0
AVOID_VEL_FWD   = 2.0

SAFE_DISTANCE   = 4.0
GAP_CLEAR       = SAFE_DISTANCE + 0.23
MIN_GAP_DEG     = 15.0
CONTROL_HZ      = 20
STUCK_WIN       = 2.0  # 2 sn
STUCK_THRESH    = 0.2  # <0.2m ilerleme => stuck

ESC_ALT         = 5.0
EMERGENCY_ASCEND= 3.0  # collision anında

# Fusion grid boyutu (fusion.py ile eşleşmeli)
FUSION_GRID_SIZE = 200
FUSION_CELL_SIZE = 0.1   # => 20m x 20m

PLAN_FORWARD_METERS = 4.0
PLAN_FORWARD_CELLS  = int(PLAN_FORWARD_METERS / FUSION_CELL_SIZE)

WAYPOINT_THRESHOLD   = 0.5  # (m) waypoint'e yaklaşım eşiği


# -----------------------------------------------------------------------------
# Yardımcı Fonksiyonlar
# -----------------------------------------------------------------------------
def yaw_wrap(deg: float) -> float:
    return (deg + 360) % 360

def polar_histogram(pts: np.ndarray) -> list[float]:
    """360° histogram: her derece için min mesafe."""
    angles = np.degrees(np.arctan2(pts[:,1], pts[:,0]))
    dists  = np.linalg.norm(pts[:,:2], axis=1)
    hist = np.full(360, np.inf)
    for ang, dist in zip(angles, dists):
        idx = int(yaw_wrap(ang))
        if dist < hist[idx]:
            hist[idx] = dist
    return hist.tolist()


def bfs_distance_transform(grid: np.ndarray) -> np.ndarray:
    """
    2D grid (0=free, 1=obs) -> en yakın engel mesafesi (piksel).
    Basit BFS. (scipy/cv2 varsa oradaki distanceTransform da kullanılabilir.)
    """
    h, w = grid.shape
    distmap = np.full((h,w), np.inf, dtype=np.float32)
    from collections import deque
    q = deque()

    # Engellerden başla (dist=0), free alanlara yayıl
    for r in range(h):
        for c in range(w):
            if grid[r,c] == 1:
                distmap[r,c] = 0
                q.append((r,c))

    dirs = [(1,0), (-1,0), (0,1), (0,-1)]
    while q:
        r, c = q.popleft()
        base_d = distmap[r,c]
        for d in dirs:
            nr, nc = r+d[0], c+d[1]
            if 0<=nr<h and 0<=nc<w:
                nd = base_d + 1
                if nd < distmap[nr,nc]:
                    distmap[nr,nc] = nd
                    q.append((nr,nc))
    return distmap

def local_astar(grid: np.ndarray, start_ij, goal_ij) -> list[tuple[int,int]]|None:
    """4-komşu A* (0=free, 1=obs)."""
    if start_ij == goal_ij:
        return [start_ij]
    rows, cols = grid.shape
    def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    visited = set()
    cost = {start_ij: 0}
    pq = []
    heapq.heappush(pq, (h(start_ij,goal_ij), 0, start_ij))
    came_from = {start_ij: None}
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]

    while pq:
        f,g,cur = heapq.heappop(pq)
        if cur == goal_ij:
            path=[cur]
            while came_from[path[-1]] is not None:
                path.append(came_from[path[-1]])
            path.reverse()
            return path
        if cur in visited:
            continue
        visited.add(cur)

        for d in dirs:
            nx = cur[0]+d[0]
            ny = cur[1]+d[1]
            if 0<=nx<rows and 0<=ny<cols and grid[nx,ny]==0:
                newg = g+1
                if (nx,ny) not in cost or newg<cost[(nx,ny)]:
                    cost[(nx,ny)] = newg
                    f2 = newg + h((nx,ny), goal_ij)
                    came_from[(nx,ny)] = cur
                    heapq.heappush(pq, (f2, newg, (nx,ny)))
    return None

def local_plan_largest_free_space(grid: np.ndarray, start_ij, max_candidates=5) -> list[tuple[int,int]]|None:
    """
    1) distance transform -> en büyük mesafeli free hücreleri sırala
    2) ilk bulduğun A* patikasını döndür
    """
    h, w = grid.shape
    distmap = bfs_distance_transform(grid)
    flat = distmap.ravel()
    idx_sorted = np.argsort(flat)[::-1]  # büyükten küçüğe
    tried = 0

    for idx in idx_sorted:
        r = idx // w
        c = idx % w
        if distmap[r,c] < 1:
            break  # engel dibindedir
        path = local_astar(grid, start_ij, (r,c))
        if path:
            return path
        tried += 1
        if tried >= max_candidates:
            break
    return None

def local_plan_multi_goals(grid: np.ndarray, start_ij) -> list[tuple[int,int]]|None:
    """
    'multi-goals': ileri, sağ-ileri, sol-ileri, sağ, sol...
    İlk bulduğumuz patikayı döndürür.
    """
    size = grid.shape[0]
    cx, cy = start_ij
    fwd = PLAN_FORWARD_CELLS
    goals = [
        (cx - fwd, cy),         # ileri
        (cx - fwd, cy + 10),    # ileri + sağ
        (cx - fwd, cy - 10),    # ileri + sol
        (cx,      cy + fwd),    # sağ
        (cx,      cy - fwd),    # sol
    ]
    for g in goals:
        if not (0<=g[0]<size and 0<=g[1]<size):
            continue
        path = local_astar(grid, start_ij, g)
        if path and len(path)>=2:
            return path
    return None

# -----------------------------------------------------------------------------
# Ana Kaçınma Sınıfı
# -----------------------------------------------------------------------------
class Explorer:
    def __init__(self, minutes: int, lidar_name: str):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, VEHICLE)

        self.deadline = time.time() + minutes*60
        self.lidar = lidar_name

        # path takibi için
        self.current_path: list[tuple[int,int]]|None = None
        self.path_index = 0

        self.fail_count = 0
        self.pos_hist = deque(maxlen=int(STUCK_WIN*CONTROL_HZ))

        # fusion başlat
        start_fusion_thread(view=False,
                            mode="fixed",   # depthcam/scaled
                            depth=20.0,
                            h_fov=90,
                            do_morph=True,  # yaprak vb. gürültüsünü azaltmak
                            morph_size=3
                            )

    def takeoff(self):
        print("[INFO] Taking off ...")
        self.client.armDisarm(True, VEHICLE)
        self.client.takeoffAsync(vehicle_name=VEHICLE).join()
        self.client.moveToZAsync(-TAKEOFF_ALT,3,vehicle_name=VEHICLE).join()

    def land(self):
        print("[INFO] Landing ...")
        self.client.hoverAsync(vehicle_name=VEHICLE).join()
        self.client.landAsync(vehicle_name=VEHICLE).join()
        self.client.armDisarm(False, VEHICLE)
        self.client.enableApiControl(False, VEHICLE)

    # Lidar ham nokta
    def get_points(self)-> np.ndarray|None:
        data = self.client.getLidarData(lidar_name=self.lidar, vehicle_name=VEHICLE)
        if len(data.point_cloud)<3:
            return None
        pts = np.array(data.point_cloud, dtype=np.float32).reshape(-1,3)
        pts = pts[~np.isnan(pts).any(axis=1)]
        return pts

    # Stuck
    def progress(self)-> float:
        if len(self.pos_hist)<2:
            return 999.0
        p0 = np.array([self.pos_hist[0].x_val, self.pos_hist[0].y_val])
        p1 = np.array([self.pos_hist[-1].x_val, self.pos_hist[-1].y_val])
        return float(np.linalg.norm(p1 - p0))

    def closest_dist(self, pts: np.ndarray)-> float:
        d = np.linalg.norm(pts[:,:2], axis=1)
        return float(d.min())

    def get_fusion_grid(self)-> np.ndarray|None:
        return get_latest_fused_grid(copy=True)

    # -------------------------------------------------------------------------
    # run: ana döngü
    # -------------------------------------------------------------------------
    def run(self):
        self.takeoff()
        dt = 1.0 / CONTROL_HZ
        yaw_deg = random.uniform(0,360)
        forward_vel = CRUISE_VEL_FWD

        try:
            while time.time()< self.deadline:
                tick = time.time()
                st = self.client.getMultirotorState(VEHICLE)
                self.pos_hist.append(st.kinematics_estimated.position)

                # 1) Çarpışma var mı?
                collision_info = self.client.simGetCollisionInfo(vehicle_name=VEHICLE)
                if collision_info.has_collided:
                    print("[COLLISION] => emergency escape!")
                    self.emergency_escape(yaw_deg)
                    # Varolan path iptal
                    self.current_path = None
                    self.fail_count=0
                    continue

                # 2) Eğer aktif bir path varsa, onu takip etmeye çalış
                if self.current_path is not None:
                    # path follow
                    done = self.update_path_following(st)
                    if not done:
                        # path takibine devam ediliyor
                        # bir "danger" vs. durumu yoksa plan/gap safhasına girmeyeceğiz
                        pass
                    else:
                        # path tamamlandı
                        self.current_path = None

                # 3) Path yoksa => plan/gap/escalate
                if self.current_path is None:
                    # danger/stuck?
                    pts = self.get_points()
                    stuck = (self.progress()<STUCK_THRESH)
                    danger = False
                    if pts is not None and pts.shape[0]>0:
                        if self.closest_dist(pts) < SAFE_DISTANCE:
                            danger = True
                    if danger or stuck:
                        print("[WARN] Danger or stuck => plan or gap.")
                        self.client.cancelLastTask(VEHICLE)
                        self.client.moveByVelocityBodyFrameAsync(0,0,0,0.1,vehicle_name=VEHICLE)

                        fused = self.get_fusion_grid()
                        found_path = None
                        if fused is not None:
                            found_path = self.local_plan_smart(fused)

                        if found_path:
                            print("[INFO] path found => start path follow")
                            self.current_path = found_path
                            self.path_index   = 0
                            self.fail_count   = 0
                        else:
                            # gap-based
                            gap_yaw = None
                            if pts is not None and pts.shape[0]>0:
                                gap_yaw = self.polar_gap(pts)

                            if gap_yaw is not None:
                                print(f"[INFO] gap-based => yaw={gap_yaw:.1f}")
                                self.fail_count=0
                                self.client.rotateToYawAsync(gap_yaw,vehicle_name=VEHICLE).join()
                                forward_vel = AVOID_VEL_FWD
                            else:
                                self.fail_count +=1
                                print(f"[INFO] No gap => escalate lvl {self.fail_count}")
                                if self.fail_count>=3:
                                    print("[ESC] ascend + 180")
                                    self.fail_count=0
                                    new_z = st.kinematics_estimated.position.z_val - ESC_ALT
                                    self.client.moveToZAsync(new_z,2,VEHICLE).join()
                                    yaw_deg = (yaw_deg + 180)%360
                                    self.client.rotateToYawAsync(yaw_deg,VEHICLE).join()
                                else:
                                    yaw_deg = (yaw_deg + 120 + random.uniform(-30,30))%360
                                    self.client.rotateToYawAsync(yaw_deg,VEHICLE).join()
                                forward_vel = AVOID_VEL_FWD

                # 4) Normal ileri (eğer path follow devrede değilse => bodyFrame forward)
                if self.current_path is None:
                    self.client.moveByVelocityBodyFrameAsync(forward_vel, 0, 0, dt, vehicle_name=VEHICLE)
                # yoksa path_following içinde update_path_following komutu veriyor

                # sabit zaman
                elapsed = time.time()-tick
                slp = dt - elapsed
                if slp<0: slp=0
                time.sleep(slp)

        finally:
            self.land()

    # -------------------------------------------------------------------------
    # Path Takibi
    # -------------------------------------------------------------------------
    def update_path_following(self, state) -> bool:
        """Mevcut path'i takip et. True dönerse path bitti."""
        if not self.current_path or self.path_index>= len(self.current_path):
            return True  # zaten bitmiş

        # grid merkez
        size = FUSION_GRID_SIZE
        cx = cy = size//2

        # o an drone grid'in tam ortasında varsayıyoruz,
        # path hücreleri => (r,c)
        # r,c => offset (cx-r, cy-c)
        target_cell = self.current_path[self.path_index]
        dx_pix = (cx - target_cell[0])
        dy_pix = (cy - target_cell[1])
        # metre
        dx_m = dx_pix * FUSION_CELL_SIZE
        dy_m = dy_pix * FUSION_CELL_SIZE

        dist_wp = math.hypot(dx_m, dy_m)
        if dist_wp < WAYPOINT_THRESHOLD:
            # waypoint'e ulaştık => sonraki
            self.path_index +=1
            if self.path_index >= len(self.current_path):
                print("[PATH] Reached final waypoint.")
                return True
            return False
        else:
            # oraya dön ve kısa ileri hareket et
            yaw_deg = math.degrees(math.atan2(dy_m, dx_m))%360
            self.client.rotateToYawAsync(yaw_deg, vehicle_name=VEHICLE).join()

            # Danger kontrol (basit):
            pts = self.get_points()
            if pts is not None and pts.shape[0]>0:
                if self.closest_dist(pts) < SAFE_DISTANCE:
                    print("[PATH] Danger => abort path!")
                    self.current_path = None
                    self.path_index=0
                    return True

            # 1 sn ileri git (hız=2 m/s)
            self.client.moveByVelocityBodyFrameAsync(2, 0, 0, 1.0, vehicle_name=VEHICLE).join()
            return False

    # -------------------------------------------------------------------------
    # Planlama Fonksiyonları
    # -------------------------------------------------------------------------
    def local_plan_smart(self, fused_grid: np.ndarray) -> list[tuple[int,int]]|None:
        """
        1) largest_free_space
        2) bulamazsa multi_goals
        """
        size = FUSION_GRID_SIZE
        cx=cy=size//2
        start_ij = (cx, cy)

        # a) largest free space
        path_lfs = local_plan_largest_free_space(fused_grid, start_ij, max_candidates=5)
        if path_lfs:
            return path_lfs

        # b) multi-goals
        path_mg = local_plan_multi_goals(fused_grid, start_ij)
        if path_mg:
            return path_mg

        return None

    def polar_gap(self, pts: np.ndarray)-> float|None:
        hist = polar_histogram(pts)
        free = [(d>GAP_CLEAR) for d in hist]
        best_start=None; run_len=0; best_len=0; best_mid=None

        for i,ok in enumerate(free+free):
            if ok:
                run_len +=1
                if run_len==1:
                    best_start=i
                if run_len>= MIN_GAP_DEG and run_len>best_len:
                    best_len= run_len
                    best_mid= best_start + run_len//2
            else:
                run_len=0

        if best_mid is None:
            return None
        return best_mid%360

    def emergency_escape(self, yaw_deg: float):
        print("[EMERGENCY] Hover -> ascend -> rotate -> forward")
        self.client.hoverAsync(vehicle_name=VEHICLE).join()
        st = self.client.getMultirotorState(VEHICLE)
        z = st.kinematics_estimated.position.z_val
        new_z = z - EMERGENCY_ASCEND
        self.client.moveToZAsync(new_z, 2, vehicle_name=VEHICLE).join()
        yaw_deg = (yaw_deg + 90 + random.uniform(-30,30))%360
        self.client.rotateToYawAsync(yaw_deg, vehicle_name=VEHICLE).join()
        self.client.moveByVelocityBodyFrameAsync(2, 0, 0, 1.0, vehicle_name=VEHICLE).join()

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=5, help="Kaç dakika uçulacak")
    ap.add_argument("--lidar", default="Lidar1", help="Lidar ismi")
    args = ap.parse_args()

    Explorer(args.minutes, args.lidar).run()


if __name__=="__main__":
    main()
