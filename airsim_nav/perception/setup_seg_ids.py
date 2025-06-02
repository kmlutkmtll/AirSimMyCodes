# setup_seg_ids.py
import airsim, re
c = airsim.MultirotorClient(); c.confirmConnection()

# Arka planı 0 yap
c.simSetSegmentationObjectID("Road.*", 0, True)
c.simSetSegmentationObjectID("Sky.*",  0, True)

# Tüm binalar 50, ağaçlar 60, araçlar 70
c.simSetSegmentationObjectID("Building.*", 50, True)
c.simSetSegmentationObjectID("Tree.*",     60, True)
c.simSetSegmentationObjectID("Vehicle.*",  70, True)
print("Segmentation IDs updated → Building=50  Tree=60  Vehicle=70")
