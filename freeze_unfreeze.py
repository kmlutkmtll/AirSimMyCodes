import os
import cv2
import numpy as np
import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()

clear_path = "dataset/clear_fog(0.5)_pair/clear"
fog_path = "dataset/clear_fog(0.5)_pair/fog"
os.makedirs(clear_path, exist_ok=True)
os.makedirs(fog_path, exist_ok=True)

def set_fog_and_refresh(fog_level, delay=0.15, refresh_frames=10):
    """Fog seviyesi ayarlanır ve efektin sahneye oturması için görüntü istenir."""
    client.simEnableWeather(True)
    client.simSetWeatherParameter(airsim.WeatherParameter.Fog, fog_level)
    for _ in range(refresh_frames):
        _ = client.simGetImages([
            airsim.ImageRequest("bottom_center", airsim.ImageType.Scene, False, False)
        ])
        time.sleep(delay)

def capture_image(save_dir, index, fog=False):
    responses = client.simGetImages([
        airsim.ImageRequest("bottom_center", airsim.ImageType.Scene, False, False)
    ])
    if not responses or responses[0].height == 0:
        print(f"[{index:03d}] ❌ Görüntü alınamadı.")
        return False

    response = responses[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)

    filename = os.path.join(save_dir, f"{index:03d}.png")
    cv2.imwrite(filename, img_rgb)
    print(f"[{index:03d}] ✅ {'Fog' if fog else 'Clear'} görüntü kaydedildi: {filename}")
    return True

def freeze_scene(): client.simPause(True)
def unfreeze_scene(): client.simPause(False)

# Başlangıç: sis kapalı, sahne serbest
set_fog_and_refresh(fog_level=0.0, refresh_frames=15)
unfreeze_scene()

for i in range(500):
    print(f"\n[{i:03d}] Görüntü çifti alınıyor...")

    # 1. Ortamı dondur
    freeze_scene()

    # 2. CLEAR: sis kapatılır ve buffer temizlenir
    set_fog_and_refresh(fog_level=0.0, refresh_frames=15)
    capture_image(clear_path, i, fog=False)

    # 3. FOG: sis aktif edilir ve tam yerleşmesi sağlanır
    set_fog_and_refresh(fog_level=0.5, refresh_frames=15)
    capture_image(fog_path, i, fog=True)

    # 4. Sahne açılır, sis temizlenir
    set_fog_and_refresh(fog_level=0.0, refresh_frames=5)
    unfreeze_scene()
    time.sleep(3.0)

print("✅ Tüm görüntü çiftleri başarıyla ve net hava farklarıyla üretildi.")
