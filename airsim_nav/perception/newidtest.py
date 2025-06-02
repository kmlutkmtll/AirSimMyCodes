import airsim
import cupy as cp

cli = airsim.MultirotorClient(); cli.confirmConnection()
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

# Kanal belirleme (RGB/RGBA ise)
if bpp >= 3:
    img = buf.reshape(h, w, bpp)
    chan = int(cp.argmax(img.sum(axis=(0, 1))).item())
    ids = img[:, :, chan]
else:
    ids = buf.reshape(h, w)

sky_id = int(cp.median(ids[:h//3, w//2 - 10:w//2 + 10]).item())
ground_id = int(cp.median(ids[-h//3:, w//2 - 10:w//2 + 10]).item())

print(f"Gökyüzü ID: {sky_id}")
print(f"Zemin ID: {ground_id}")
