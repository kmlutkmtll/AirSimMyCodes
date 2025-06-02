#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Lidar Test with AirSim – Real-time Occupancy Grid
"""

import airsim
import time
import numpy as np
from lidar_processor_gpu import occupancy_grid, _GPU, xp

# AirSim bağlantısı
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# Lidar adı
lidar_name = "Lidar1"

# Frame hızı
hz = 5.0
dt = 1.0 / hz

print("GPU mode:", _GPU)
print("Başlıyoruz... (Çıkmak için Ctrl+C)")

try:
    while True:
        start = time.time()

        # Lidar verisini al
        lidar_data = client.getLidarData(lidar_name=lidar_name)
        if not lidar_data.point_cloud:
            print("[Uyarı] Lidar verisi yok.")
            time.sleep(dt)
            continue

        # Nokta bulutunu işle
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        grid = occupancy_grid(points)

        if _GPU:
            xp.cuda.Stream.null.synchronize()

        occ_percent = float(grid.mean() * 100)
        print(f"[{lidar_data.time_stamp}] Grid occupancy: {occ_percent:.2f}%")

        elapsed = time.time() - start
        if elapsed < dt:
            time.sleep(dt - elapsed)

except KeyboardInterrupt:
    print("Durduruluyor...")

finally:
    client.armDisarm(False)
    client.enableApiControl(False)
