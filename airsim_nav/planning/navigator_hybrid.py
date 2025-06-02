# ────────────────────────────────────────────────────────────────────────────
# navigator_hybrid.py
# ────────────────────────────────────────────────────────────────────────────
"""AirSim demo: Hibrit kaçınma"""

import airsim, math, time, threading
from airsim_nav.config import Params
from airsim_nav.mapping.fusion_gpu import start_fusion_thread
from airsim_nav.planning.planner_hybrid import HybridPlanner

if __name__=="__main__":
    cfg=Params.load(); cli=airsim.MultirotorClient(); cli.confirmConnection()
    cli.enableApiControl(True); cli.armDisarm(True)
    cli.takeoffAsync().join(); cli.moveToZAsync(-cfg.takeoff_alt,1).join()

    start_fusion_thread(view=False,mode="fixed")
    planner=HybridPlanner(cfg); HZ=cfg.control_hz

    while True:
        yaw=airsim.to_eularian_angles(cli.getMultirotorState().kinematics_estimated.orientation)[2]
        vx,vy,w=planner.step(yaw)
        cli.moveByVelocityAsync(vx,vy,0,1/HZ,yaw_mode=airsim.YawMode(True,math.degrees(w)))
        time.sleep(1/HZ)