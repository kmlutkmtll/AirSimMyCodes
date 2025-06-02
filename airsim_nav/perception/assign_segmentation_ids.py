import airsim
import cupy as cp

# AirSim client
cli = airsim.MultirotorClient()
cli.confirmConnection()

# ────────────── 1. Sabit ID Atama ──────────────
# Gökyüzü → 255
cli.simSetSegmentationObjectID("Sky.*", 255, True)
# Tüm zemin (ground) objeleri → 0
cli.simSetSegmentationObjectID("Landscape.*", 0, True)


print("[✓] Gökyüzü ID → 255, Zemin ID → 0 olarak atandı.")

# ────────────── 2. Segmentasyon Görüntüsü Al ──────────────
rsp = cli.simGetImages([
    airsim.ImageRequest(
        camera_name="front_center",
        image_type=airsim.ImageType.Segmentation,
        pixels_as_float=False,
        compress=False
    )
], vehicle_name="Drone1")[0]

w, h = rsp.width, rsp.height
buf = cp.frombuffer(rsp.image_data_uint8, dtype=cp.uint8)
bpp = buf.size // (w * h)

# Kanal ayrımı (RGB ise)
if bpp >= 3:
    img = buf.reshape(h, w, bpp)
    chan = int(cp.argmax(img.sum(axis=(0, 1))).item())
    ids = img[:, :, chan]
else:
    ids = buf.reshape(h, w)

unique_ids = cp.unique(ids)
print(f"[🧠] Görüntüdeki aktif segmentasyon ID'leri: {cp.asnumpy(unique_ids).tolist()}")

# Gökyüzü ve zemin örnek değerleri (ortalama merkez bölge)
sky_id = int(cp.median(ids[:h//3, w//2 - 10:w//2 + 10]).item())
ground_id = int(cp.median(ids[-h//3:, w//2 - 10:w//2 + 10]).item())

print(f"[🔎] Tespit edilen Gökyüzü ID: {sky_id}")
print(f"[🔎] Tespit edilen Zemin ID: {ground_id}")
