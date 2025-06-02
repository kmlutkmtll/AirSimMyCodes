#!/usr/bin/env python3
import numpy as np, torch, time, airsim
from airsim_nav.perception.segmentation_processor_gpu import fetch_mask, _LUT_GPU

cli = airsim.MultirotorClient(); cli.confirmConnection()

# ----- gerçek AirSim süresi -----
t0 = time.time()
mask = fetch_mask(cli)                    # tam yol (simGetImages + GPU)
torch.cuda.synchronize()
print(f"Full call : {(time.time()-t0)*1e3:.2f} ms")

# ----- saf GPU yolunu ölç -----
rsp = cli.simGetImages([airsim.ImageRequest(
        "front_center", airsim.ImageType.Segmentation,
        pixels_as_float=False, compress=False)])[0]

h, w  = rsp.height, rsp.width
buf   = np.frombuffer(rsp.image_data_uint8, np.uint8)
bpp   = buf.size // (h*w)                 # 1,3 veya 4

ids_t = torch.frombuffer(buf, dtype=torch.uint8).to("cuda")
ids_t = ids_t.view(h, w) if bpp == 1 else ids_t.view(h, w, bpp)[:, :, 0]

evt0, evt1 = torch.cuda.Event(True), torch.cuda.Event(True)
evt0.record()
mask_t = _LUT_GPU[ids_t.to(torch.long)]
evt1.record(); evt1.synchronize()
print(f"Pure LUT kernel: {evt0.elapsed_time(evt1):.3f} ms")
