#!/usr/bin/env python3
"""
AirSim → ChaIR‑Tiny (DehazeModel) → YOLOv11‑m (Ultralytics)
----------------------------------------------------------------
• Dehaze edilmiş kare, YOLOv11‑m'e verilir.
• YOLO giriş boyutu (imgsz) stride 32'nin katı olan **384**.
• Uyarılar kalktı; model GPU'da, .fuse() hatası giderildi.

Çalıştırma:
    python airsim_dehaze_yolo11m.py --host 127.0.0.1 --cam front_center
"""
from __future__ import annotations
import os, sys, time, argparse, cv2, airsim, numpy as np, torch
from pathlib import Path
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Yol ve model yüklemeleri
# ----------------------------------------------------------------------------
ROOT_DEHAZE = Path(r"C:/AirSim/MyCodes/EnhancementModels/ChaIR")
CKPT_DEHAZE = ROOT_DEHAZE / "weights" / "ots_38.01.pkl"
CKPT_YOLO   = Path(r"C:/AirSim/MyCodes/full_pipeline/weights/yolo11m.pt")

sys.path.append(str(ROOT_DEHAZE / "Dehazing" / "OTS" / "models" / "utils"))
from EnhancementModels.ChaIR.Dehazing.OTS.models.utils.infer_chair_tiny import DehazeModel

print("[Load] ChaIR‑Tiny DehazeModel →", CKPT_DEHAZE)
model_dehaze = DehazeModel(str(CKPT_DEHAZE), device="cuda")

print("[Load] YOLOv11‑m →", CKPT_YOLO)
model_yolo = YOLO(str(CKPT_YOLO))          # Ultralytics wrapper
# CONV+BN katmanlarını birleştir (fuse) — .model üzerinde çalışır
if hasattr(model_yolo, "model") and hasattr(model_yolo.model, "fuse"):
    model_yolo.model.fuse()
CONF_THRES   = 0.25
IMGSZ        = 384              # stride 32'ye tam bölünür → uyarı yok
H_CAM, W_CAM = 360, 640         # AirSim kamera çözünürlüğü (değiştirmeyin)

# ----------------------------------------------------------------------------
# Yardımcı: YOLO çıktısını çiz
# ----------------------------------------------------------------------------
COL = (0, 255, 0)

def draw_detections(img: np.ndarray, results) -> np.ndarray:
    if not results:
        return img
    r = results[0]
    if r.boxes is None:
        return img
    names = r.names
    for box in r.boxes:
        conf = float(box.conf)
        if conf < CONF_THRES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = names[int(box.cls)]
        cv2.rectangle(img, (x1, y1), (x2, y2), COL, 2)
        cv2.putText(img, f"{cls} {conf:.2f}", (x1, max(12, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL, 1, cv2.LINE_AA)
    return img

# ----------------------------------------------------------------------------
# Ana döngü
# ----------------------------------------------------------------------------

def run(host: str, cam: str):
    cli = airsim.MultirotorClient(ip=host)
    cli.confirmConnection(); print("[AirSim] Connected →", host)
    req = airsim.ImageRequest(cam, airsim.ImageType.Scene, pixels_as_float=False, compress=False)

    frame_cnt, t0 = 0, time.time()
    print("[Info] Başladı — Q ile çıkış")
    while True:
        resp = cli.simGetImages([req])[0]
        if resp.width == 0:
            continue
        bgr = np.frombuffer(resp.image_data_uint8, np.uint8).reshape(resp.height, resp.width, 3)
        if (resp.width, resp.height) != (W_CAM, H_CAM):
            bgr = cv2.resize(bgr, (W_CAM, H_CAM), interpolation=cv2.INTER_LINEAR)

        # 1) Dehaze (BGR→BGR)
        clean = model_dehaze(bgr)

        # 2) YOLO (letter‑boxed, device=GPU)
        results = model_yolo.predict(clean, imgsz=IMGSZ, conf=CONF_THRES, device=0, verbose=False)

        # 3) Kutuları çiz & göster
        out = draw_detections(clean.copy(), results)
        cv2.imshow("Dehaze + YOLOv11‑m", out)
        if cv2.waitKey(1) & 0xFF in (27, ord('q'), ord('Q')):
            break

        frame_cnt += 1
        if frame_cnt == 120:
            fps = 120 / (time.time() - t0)
            print(f"[Perf] 120 kare → {fps:.1f} FPS")
            frame_cnt = 0; t0 = time.time()

    cv2.destroyAllWindows()

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="AirSim IP")
    ap.add_argument("--cam",  default="bottom_center", help="AirSim camera name")
    args = ap.parse_args()
    run(args.host, args.cam)