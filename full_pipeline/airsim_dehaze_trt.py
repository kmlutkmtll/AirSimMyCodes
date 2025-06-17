#!/usr/bin/env python3
"""
AirSim ► ChaIR-Tiny (FP16 TensorRT) ► YOLOv11-m (.pt, Ultralytics)
=================================================================
• Dehaze motoru: TensorRT 8-10 uyumlu, FP16.
• YOLOv11-m: PyTorch / Ultralytics, GPU'da (TRT değil).
• YOLO giriş boyutu stride 32’nin katı (512) → uyarı yok.
"""

from __future__ import annotations
import time, argparse, cv2, airsim, numpy as np
from pathlib import Path

# ───────────── Ultralytics (PyTorch) ─────────────
from ultralytics import YOLO          # pip install ultralytics>=8.1

# ───────────── TensorRT / CUDA (Dehaze) ──────────
import tensorrt as trt
import pycuda.driver as cuda; import pycuda.autoinit   # noqa: F401

# ───────────────── Config ────────────────────────
DEHAZE_ENGINE = Path(r"C:/AirSim/MyCodes/full_pipeline/weights/dehaze_chair_tiny_fp16.engine")
YOLO_PT      = Path(r"C:/AirSim/MyCodes/full_pipeline/weights/yolo11m.pt")

CONF_THRES   = 0.25
IMGSZ        = 512                 # stride 32'nin katı ⇒ warning yok
H_CAM, W_CAM = 360, 640            # AirSim kamera
CLR_BOX      = (0, 255, 0)

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# ───────────── TensorRT runner (aynen bıraktın) ─────────────
class TrtRunner:
    def __init__(self, engine_path: str | Path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context   = self.engine.create_execution_context()
        self.stream    = cuda.Stream()

        self.inputs  = [n for n in self.engine if self.engine.get_tensor_mode(n)==trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.engine if self.engine.get_tensor_mode(n)==trt.TensorIOMode.OUTPUT]
        self.hbuf, self.dbuf, self.shape = {}, {}, {}

        for n in (*self.inputs, *self.outputs):
            dtype  = trt.nptype(self.engine.get_tensor_dtype(n))
            shape  = tuple(max(1,d) for d in self.context.get_tensor_shape(n))  # dinamik? sabitle
            host   = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dev    = cuda.mem_alloc(host.nbytes)
            self.hbuf[n], self.dbuf[n], self.shape[n] = host, dev, shape
            self.context.set_tensor_address(n, int(dev))

    def infer(self, inp: np.ndarray) -> np.ndarray:
        ni, no = self.inputs[0], self.outputs[0]
        np.copyto(self.hbuf[ni], inp.ravel())
        cuda.memcpy_htod_async(self.dbuf[ni], self.hbuf[ni], self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.hbuf[no], self.dbuf[no], self.stream)
        self.stream.synchronize()
        return self.hbuf[no].reshape(self.shape[no])

# ───────────── Yardımcılar ─────────────
def prep_dehaze(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 ► NCHW FP16 RGB [0-1]"""
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2,0,1)[None]).astype(np.float16)

def post_dehaze(nchw: np.ndarray) -> np.ndarray:
    """NCHW FP16 RGB [0-1] ► BGR uint8"""
    rgb = np.clip(nchw[0].transpose(1,2,0)*255.0, 0, 255).astype(np.uint8)
    return rgb[..., ::-1]

def draw_dets(img: np.ndarray, results) -> np.ndarray:
    res = results[0]
    if not res.boxes:                          # YOLO 8 API
        return img
    names = res.names
    for b in res.boxes:
        conf = float(b.conf)
        if conf < CONF_THRES: continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cls = names[int(b.cls)]
        cv2.rectangle(img, (x1,y1), (x2,y2), CLR_BOX, 2)
        cv2.putText(img, f"{cls} {conf:.2f}", (x1, max(12,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_BOX, 1, cv2.LINE_AA)
    return img

# ───────────── Ana döngü ─────────────
def run(host: str, cam: str, show: bool):
    print("[Load] Dehaze TRT  →", DEHAZE_ENGINE)
    dehaze = TrtRunner(DEHAZE_ENGINE)

    print("[Load] YOLOv11-m  →", YOLO_PT)
    yolo = YOLO(str(YOLO_PT))
    if hasattr(yolo.model, "fuse"):
        yolo.model.fuse()          # BN+Conv birleş­, uyarı kalkar
    yolo.to('cuda')
    yolo.eval()

    cli = airsim.MultirotorClient(ip=host)
    cli.confirmConnection(); print("[AirSim] Connected")

    req = airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False)
    t0, fcnt = time.time(), 0

    while True:
        r = cli.simGetImages([req])[0]
        if r.width == 0: continue
        frame = np.frombuffer(r.image_data_uint8, np.uint8).reshape(r.height, r.width, 3)
        if (r.height, r.width) != (H_CAM, W_CAM):
            frame = cv2.resize(frame, (W_CAM, H_CAM), cv2.INTER_LINEAR)

        # 1) Dehaze (TRT)
        clean = post_dehaze(dehaze.infer(prep_dehaze(frame)))

        # 2) YOLO (PyTorch)
        results = yolo.predict(clean, imgsz=IMGSZ, conf=CONF_THRES,
                               device=0, verbose=False)

        # 3) Çiz & göster
        out = draw_dets(clean.copy(), results)
        if show:
            cv2.imshow("Dehaze + YOLOv11-m", out)
            if cv2.waitKey(1) & 0xFF in (27, ord('q'), ord('Q')):
                break

        fcnt += 1
        if fcnt == 120:
            fps = 120 / (time.time() - t0)
            print(f"[Perf] 120 kare → {fps:.1f} FPS")
            fcnt, t0 = 0, time.time()

    cv2.destroyAllWindows()

# ───────────── CLI ─────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="AirSim IP")
    ap.add_argument("--cam",  default="bottom_center", help="AirSim camera name")
    ap.add_argument("--display", action="store_true", help="OpenCV penceresi")
    args = ap.parse_args()
    run(args.host, args.cam, args.display)
