import airsim
import numpy as np
import time
import threading
import keyboard  # pip install keyboard

# AirSim'e Bağlan
client = airsim.MultirotorClient()
client.confirmConnection()

vehicle_name = "Drone1"

# Drone'u Hazırlama
client.enableApiControl(True, vehicle_name=vehicle_name)
client.armDisarm(True, vehicle_name=vehicle_name)

print("[INFO] Kalkış yapılıyor...")
client.takeoffAsync(vehicle_name=vehicle_name).join()
client.moveToZAsync(-5, 3, vehicle_name=vehicle_name).join()  # 5 metre havalan

# Global değişkenler
camera_pitch = 0.0
camera_yaw = 0.0
camera_roll = 0.0

velocity = 3  # Drone hızı (m/s)
duration = 0.1  # Komut süresi (saniye)

def control_drone():
    global camera_pitch, camera_yaw

    sensitivity = 5 * np.pi / 180  # 5 derece hassasiyet (daha iyi kontrol için)

    while True:
        vx, vy, vz = 0, 0, 0
        yaw_rate = 0

        # Drone Hareket Kontrolleri
        if keyboard.is_pressed('w'):
            vx = velocity
        if keyboard.is_pressed('s'):
            vx = -velocity
        if keyboard.is_pressed('a'):
            vy = -velocity
        if keyboard.is_pressed('d'):
            vy = velocity
        if keyboard.is_pressed('q'):
            vz = -velocity
        if keyboard.is_pressed('e'):
            vz = velocity

        # Drone Yaw Dönme
        if keyboard.is_pressed('z'):
            yaw_rate = -30
        if keyboard.is_pressed('c'):
            yaw_rate = 30
        # Drone hareket ettir
        client.moveByVelocityAsync(vx, vy, vz, duration,
                                   yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
                                   vehicle_name=vehicle_name)

        # ✅ KAMERA KONTROLÜ
        # i: yukarı bak, k: aşağı bak
        if keyboard.is_pressed('i'):  # Kamera yukarı eğ
            camera_pitch += sensitivity
        if keyboard.is_pressed('k'):  # Kamera aşağı eğ
            camera_pitch -= sensitivity
        if keyboard.is_pressed('j'):  # Kamera sola çevir
            camera_yaw -= sensitivity
        if keyboard.is_pressed('l'):  # Kamera sağa çevir
            camera_yaw += sensitivity

        # ✅ Limit veriyoruz (pitch sınırlı, yaw sınırsız dönebilir)
        camera_pitch = np.clip(camera_pitch, -np.pi / 2, np.pi / 2)

        # ✅ Kamera yönünü uygula (kamera adı front_center)
        pose = airsim.Pose(airsim.Vector3r(0, 0, 0),
                           airsim.to_quaternion(camera_pitch, camera_roll, camera_yaw))
        client.simSetCameraPose("front_center", pose, vehicle_name=vehicle_name)

        time.sleep(0.05)

        ld = client.getLidarData("Lidar1")
        print("pts:", len(ld.point_cloud) // 3)

# Kontrol Thread'i Başlat
control_thread = threading.Thread(target=control_drone)
control_thread.daemon = True
control_thread.start()


print("[INFO] Klavye kontrolü aktif! ESC tuşuna basarak çıkabilirsin.")

# Sonsuz döngü → ESC ile çık
try:
    while True:
        if keyboard.is_pressed('esc'):
            print("[INFO] Çıkış yapılıyor...")
            break
        time.sleep(0.1)

except KeyboardInterrupt:
    pass

# Drone İnişi ve Temizlik
client.hoverAsync(vehicle_name=vehicle_name).join()
client.landAsync(vehicle_name=vehicle_name).join()
client.armDisarm(False, vehicle_name=vehicle_name)
client.enableApiControl(False, vehicle_name=vehicle_name)

ld = client.getLidarData("Lidar1")
print("pts:", len(ld.point_cloud)//3)

print("[INFO] Görev tamamlandı.")
