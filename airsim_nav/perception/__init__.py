# Auto-switch to GPU implementation if CUDA mevcut
try:
    from .lidar_processor_gpu import occupancy_grid_gpu as occupancy_grid
except (ImportError, RuntimeError):
    # CUDA yoksa veya Torch yanlış derlenmişse güvenli geri dönüş
    from .lidar_processor import occupancy_grid

try:
    from .segmentation_processor_gpu import fetch_mask as fetch_mask
except (ImportError, RuntimeError):
    from .segmentation_processor import fetch_mask

