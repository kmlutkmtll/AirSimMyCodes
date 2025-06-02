
from airsim_nav.config import Params
from airsim_nav.perception.segmentation_processor_gpu import fetch_mask
import numpy as np

cfg = Params.load("tests/orient_test.yaml")
# 3×3 sentetik RGBA (bir köşede kırmızı) → yönü değiştirdi mi bakacağız
raw = np.zeros((3,3,4), np.uint8); raw[0,0]=[255,0,0,255]

mask = fetch_mask(None, "front_center", "Drone1",
                  ids={1}, params=cfg, _raw_override=raw)

print(mask)        # beklenen: 180° döndüğü için kırmızı köşe (2,2)

