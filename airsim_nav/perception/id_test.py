# id_inspect.py (gelişmiş)
import airsim, numpy as np, cv2, os

# ➊ AirSim'e bağlan
cl = airsim.MultirotorClient()
cl.confirmConnection()

# ➋ Segmentasyon görüntüsünü al
rsp = cl.simGetImages([airsim.ImageRequest(
    "front_center",
    airsim.ImageType.Segmentation,
    pixels_as_float=False,
    compress=False)])[0]

h, w = rsp.height, rsp.width
img = np.frombuffer(rsp.image_data_uint8, np.uint8).reshape(h, w, 3)

# ➌ Görüntüdeki benzersiz ID’leri al
ids, cnt = np.unique(img[:, :, 0], return_counts=True)
print(f"[✓] Toplam {len(ids)} segment ID bulundu.")

# ➍ Maske klasörünü hazırla
os.makedirs("id_masks", exist_ok=True)

# ➎ Obje isimlerini ID'ye göre eşleştir
scene_objects = cl.simListSceneObjects("")
id_to_names = {}

for obj in scene_objects:
    try:
        oid = cl.simGetSegmentationObjectID(obj)
        if oid in ids:
            id_to_names.setdefault(oid, []).append(obj)
    except:
        continue

# ➏ Her ID için maske kaydet + isimleri yaz
for i, c in zip(ids, cnt):
    mask = (img[:, :, 0] == i).astype(np.uint8) * 255
    cv2.imwrite(f"id_masks/id_{i:03d}.png", mask)
    obj_names = id_to_names.get(i, [])
    obj_label = ", ".join(obj_names) if obj_names else "⛔ eşleşen obje yok"
    print(f"ID {i:3d}  → {c:6d} px  ({c/h/w*100:.1f}%)  →  {obj_label}")
