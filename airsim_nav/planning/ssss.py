from __future__ import annotations

import argparse, math, time, random as rnd
from collections import deque
from typing import Tuple

import numpy as np
import airsim

# ───────────────────────── PARAMETERS ─────────────────────────
VEHICLE          = "Drone1"
TAKEOFF_ALT      = 3.0   # m
MAX_VEL_FWD      = 4.0    # m/s
SIDESTEP_VEL     = 3.0    # m/s
SAFE_DISTANCE    = 4.0    # m
BRAKE_DIST       = 6.0    # m
SIDESTEP_DIST    = 3.0    # m
MIN_SIDE_CLEAR   = 4.5    # m
ESC_ALT          = 3.0    # m
CONTROL_HZ       = 20
FILTER_THRESH    = 0.25

Vec3 = Tuple[float, float, float]

# ───────────────────────── HELPERS ────────────────────────────

def yaw_wrap(deg: float) -> float:
    return (deg + 360) % 360


def polar_histogram(pts: np.ndarray) -> np.ndarray:
    ang = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    dist = np.linalg.norm(pts[:, :2], axis=1)
    hist = np.full(360, np.inf)
    for a, d in zip(ang, dist):
        i = int(yaw_wrap(a))
        if d < hist[i]:
            hist[i] = d
    return hist

# ───────────────────────── CLASS ────────────────────────────
class Explorer:
    def __init__(self, minutes: int, lidar_name: str):
        self.c = airsim.MultirotorClient()
        self.c.confirmConnection()
        self.c.enableApiControl(True, VEHICLE)
        self.end_time = time.time() + minutes * 60
        self.lidar = lidar_name
        self.cruise_yaw = rnd.uniform(0, 360)

    # ----- helpers -----
    def takeoff(self):
        self.c.armDisarm(True, VEHICLE)
        self.c.takeoffAsync(vehicle_name=VEHICLE).join()
        self.c.moveToZAsync(-TAKEOFF_ALT, 3, vehicle_name=VEHICLE).join()
        self.c.rotateToYawAsync(self.cruise_yaw, vehicle_name=VEHICLE).join()

    def land(self):
        self.c.hoverAsync(vehicle_name=VEHICLE).join()
        self.c.landAsync(vehicle_name=VEHICLE).join()
        self.c.armDisarm(False, VEHICLE)
        self.c.enableApiControl(False, VEHICLE)

    # Lidar helpers
    def get_pts(self):
        d = self.c.getLidarData(lidar_name=self.lidar, vehicle_name=VEHICLE)
        if len(d.point_cloud) < 3:
            return None
        pts = np.asarray(d.point_cloud, np.float32).reshape(-1, 3)
        pts = pts[~np.isnan(pts).any(axis=1)]
        return pts if pts.size else None

    # ----- main loop -----
    def run(self):
        self.takeoff()
        dt = 1.0 / CONTROL_HZ
        try:
            while time.time() < self.end_time:
                tic = time.time()
                speed_cmd = MAX_VEL_FWD
                pts = self.get_pts()
                if pts is not None:
                    hist = polar_histogram(pts)
                    closest = float(np.min(hist))
                    if np.isinf(closest):
                        closest = 999  # hiçbir engel yok
                    # predictive brake
                    if closest < BRAKE_DIST:
                        speed_cmd = MAX_VEL_FWD * max(0.1, (closest - SAFE_DISTANCE)/(BRAKE_DIST - SAFE_DISTANCE))
                    # emergency
                    if closest < SAFE_DISTANCE:
                        self.emergency(pts, hist)
                        speed_cmd = 0  # emergency fonksiyonu kendi hız komutunu gönderir
                # send cruise velocity
                if speed_cmd > 0:
                    self.c.moveByVelocityBodyFrameAsync(speed_cmd, 0, 0, dt,
                                                        vehicle_name=VEHICLE,
                                                        drivetrain=airsim.DrivetrainType.ForwardOnly)
                time.sleep(max(0, dt - (time.time() - tic)))
        finally:
            self.land()

    # ----- avoidance core -----
    def emergency(self, pts: np.ndarray, hist: np.ndarray):
        self.c.cancelLastTask(vehicle_name=VEHICLE)
        self.c.moveByVelocityBodyFrameAsync(0, 0, 0, 0.1, vehicle_name=VEHICLE).join()
        right_c = hist[int(yaw_wrap(self.cruise_yaw + 90))]
        left_c  = hist[int(yaw_wrap(self.cruise_yaw - 90))]
        side = 1 if right_c > left_c else -1
        if max(right_c, left_c) > MIN_SIDE_CLEAR:
            yaw = yaw_wrap(self.cruise_yaw + 90*side)
            self.c.rotateToYawAsync(yaw, vehicle_name=VEHICLE).join()
            dur = SIDESTEP_DIST / SIDESTEP_VEL
            self.c.moveByVelocityBodyFrameAsync(SIDESTEP_VEL, 0, 0, dur,
                                                vehicle_name=VEHICLE).join()
            self.c.rotateToYawAsync(self.cruise_yaw, vehicle_name=VEHICLE).join()
        else:
            state = self.c.getMultirotorState(vehicle_name=VEHICLE)
            new_z = state.kinematics_estimated.position.z_val - ESC_ALT
            self.c.moveToZAsync(new_z, 2, vehicle_name=VEHICLE).join()
            self.cruise_yaw = yaw_wrap(self.cruise_yaw + 150 + rnd.uniform(-20, 20))
            self.c.rotateToYawAsync(self.cruise_yaw, vehicle_name=VEHICLE).join()
        self.c.moveByVelocityBodyFrameAsync(SIDESTEP_VEL, 0, 0, 1.0,
                                            vehicle_name=VEHICLE).join()

# ───────────────────────── CLI ────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=5)
    ap.add_argument("--lidar", default="Lidar1")
    args = ap.parse_args()
    Explorer(args.minutes, args.lidar).run()

if __name__ == "__main__":
    main()