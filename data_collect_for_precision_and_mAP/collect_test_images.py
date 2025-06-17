import os
import cv2
import numpy as np
import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()

save_path = "clear"
os.makedirs(save_path, exist_ok=True)

for i in range(500):
    responses = client.simGetImages([
        airsim.ImageRequest("bottom_center", airsim.ImageType.Scene, False, False)
    ])

    for idx, response in enumerate(responses):
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(1440, 2560, 3)  # 2K çözünürlük
        filename = os.path.join(save_path, f"{i:03d}.png")
        cv2.imwrite(filename, img_rgb)

    print(f"Görüntü kaydedildi: {filename}")
    time.sleep(0.5)

print("Yağmurlu gündüz - 50 görüntü kaydedildi!")
