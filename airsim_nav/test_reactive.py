#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_reactive.py – 360° polar-histogram reaktif kaçınma testi
"""

import time, math, logging, airsim
from airsim_nav.mapping.fusion    import start_fusion_thread, get_latest_fused_grid
from airsim_nav.planning.reactive_navigator import ReactiveAvoider
from airsim_nav.planning.config   import params
from airsim_nav.config            import Params

# ─── log ayarı ────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("test_reactive")

# ─── global ayarlar ──────────────────────────────────
cfg  = Params.load()               # lidar_name, vehicle, takeoff_alt ...
VEH  = cfg.vehicle
DT   = params.dt_cmd

# ─── kaçınma nesnesi ────────────────────────────────
avoid = ReactiveAvoider()

def main() -> None:
    # 1) fusion thread
    start_fusion_thread(view=False)
    log.info("[fusion] thread launched")

    # 2) AirSim hazırlık
    cli = airsim.MultirotorClient(); cli.confirmConnection()
    cli.enableApiControl(True, VEH); cli.armDisarm(True, VEH)
    cli.takeoffAsync(vehicle_name=VEH).join()
    cli.moveToZAsync(-params.target_alt, 2, vehicle_name=VEH).join()
    log.info("🚁 take-off complete – reactive loop running")

    try:
        while True:
            grid = get_latest_fused_grid()
            if grid is None:
                time.sleep(0.05); continue

            # mevcut irtifa (+m)  – z_val negatif
            alt = -cli.getMultirotorState(VEH).kinematics_estimated.position.z_val

            vx, vy, vz, yaw_rate = avoid.choose_cmd(grid, alt)

            cli.moveByVelocityBodyFrameAsync(
                vx, vy, vz, DT,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(yaw_rate)),
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                vehicle_name=VEH)

            time.sleep(DT)

    except KeyboardInterrupt:
        log.info("⌫ user interrupt – hover")
    finally:
        cli.hoverAsync(vehicle_name=VEH).join()
        cli.enableApiControl(False, VEH)

if __name__ == "__main__":
    main()
