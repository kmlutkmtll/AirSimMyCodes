"""
Canlı Stencil-ID Toplayıcı
--------------------------
• front_center Segmentation görüntüsünden her karede benzersiz ID’leri toplar
• config.yaml içindeki seg_ids listesine (veya sözlüğüne) ekler
• all_ids.json: {id: toplam_piksel} biçiminde döküm bırakır
Çalıştır:
    python -m airsim_nav.tools.id_collector
Gezerken uç; Ctrl+C ile durduğunda “Building/Tree/Vehicle” gibi
kategori sorusu sorar, sonucu config.yaml’a yazar.
"""
from __future__ import annotations
from pathlib import Path
import time, json, sys, collections
import numpy as np
import airsim
from airsim_nav.config import Params

CFG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
CAM = "front_center"
VEHICLE = "Drone1"

client = airsim.MultirotorClient(); client.confirmConnection()
print("[INFO] Connected – uçarken Ctrl+C ile bitir.")

acc = collections.Counter()
t0, n = time.perf_counter(), 0
try:
    while True:
        rsp = client.simGetImages([airsim.ImageRequest(CAM,
                                                       airsim.ImageType.Segmentation,
                                                       False, False)],
                                  vehicle_name=VEHICLE)[0]
        if rsp.width == 0: time.sleep(0.05); continue
        arr = np.frombuffer(rsp.image_data_uint8, np.uint8)
        ids, cnt = np.unique(arr[::3], return_counts=True)  # R=B=G → tek kanal yeter
        for i,c in zip(ids, cnt): acc[int(i)] += int(c)
        n += 1
        if n % 60 == 0:
            fps = n / (time.perf_counter() - t0)
            print(f"\r[INFO] … topluyor ({fps:.1f} Hz)", end="", flush=True)
except KeyboardInterrupt:
    pass

print("\n-- En çok görülen 30 ID --")
for i,c in acc.most_common(30):
    pct = 100 * c / acc.total()
    print(f"ID {i:3d} → {c:7d} px ({pct:4.1f} %)")

# —— Kullanıcıdan kategori seçimi ——
cat = input("\nBu oturumdaki ID’leri hangi kategoriye eklemek istersin?\n"
            "Örn. building / tree / vehicle / other  » ").strip() or "misc"

# —— config.yaml güncelle ——
cfg = Params.load(CFG_PATH)
seg_ids = getattr(cfg, "seg_ids", {})
if isinstance(seg_ids, list):            # eski basit liste formatı
    seg_ids = {"default": seg_ids}
seg_ids.setdefault(cat, [])
for i in acc:
    if i not in seg_ids[cat] and i != 0:
        seg_ids[cat].append(i)
cfg.seg_ids = seg_ids
cfg.save(CFG_PATH)

# —— all_ids.json dump ——
with open("all_ids.json", "w") as f:
    json.dump(acc, f, indent=2)

print(f"[INFO] {len(acc)} ID toplandı → config.yaml güncellendi (kategori: {cat})")
print("all_ids.json dosyasında tam frekans listesi var.")
