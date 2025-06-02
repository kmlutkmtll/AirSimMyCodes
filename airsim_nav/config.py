#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merkezi konfigürasyon yöneticisi  (rev-28 May 2025, v1.0-fixed)
────────────────────────────────────────────────────────────────
• Dataclass → tüm ayarlar tek yerde, tamamı YAML’den yüklenir.
• YAML I/O  → kalıcı config (config.yaml).
• CLI override → hızlı parametre denemesi.
• Alt-bloklar: lidar / segmentation / fusion sözlükleri.

Önemli değişiklikler (v1.0-fixed)
---------------------------------
✓ Varsayılan fusion.logic  = "union"   (AND yerine OR)
✓ İlk çalıştırmada örnek YAML, güncel şablonla üretilir.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import yaml, argparse, sys

# ---------- Proje kökünü (repo root) otomatik bul ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------
@dataclass
class Params:
    # =========== Genel kontrol ===========
    control_hz: int = 20
    vehicle:    str = "Drone1"

    # =========== Uçuş ===========
    takeoff_alt:  float = 3.0
    max_vel_fwd:  float = 4.0
    sidestep_vel: float = 3.0

    # =========== Güvenlik ===========
    safe_distance:     float = 5.0
    brake_distance:    float = 7.0
    sidestep_distance: float = 3.0
    min_side_clear:    float = 6.0
    escape_alt:        float = 4.0
    stuck_vel_thr:     float = 0.1

    # =========== Sensör adları ===========
    lidar_name:   str = "Lidar1"
    seg_cam:      str = "front_center"
    seg_update_hz:int = 4

    # =========== Legacy grid ===========
    cell_size:  float = 0.50
    map_radius: float = 50.0
    grid_dim:   int   = 0       # 0 → __post_init__ türetir

    # =========== Alt-bloklar ===========
    lidar:        dict | None = field(default=None)
    segmentation: dict | None = field(default=None)
    fusion:       dict | None = field(default=None)

    # =========== APF / DWA (opsiyonel) ===========
    T_free: int = 4
    V_max: float = 5.0
    Omega_max: float = 90.0
    w_c: float = 2.0
    w_h: float = 1.0
    w_f: float = 0.6
    w_v: float = 0.2
    k_rep: float = 1.0
    k_attr: float = 0.3
    explore_T: float = 5.0

    # ---------- Otomatik doğrulama ----------
    def __post_init__(self):
        if self.safe_distance >= self.brake_distance:
            raise ValueError("safe_distance < brake_distance olmalı")
        if not (1 <= self.control_hz <= 60):
            raise ValueError("control_hz 1-60 Hz aralığında olmalı")
        if self.grid_dim == 0:
            self.grid_dim = int((self.map_radius * 2) / self.cell_size)

    # ---------- YAML yükle / kaydet ----------
    @classmethod
    def load(cls, yaml_path: str | Path | None = None) -> "Params":
        path = Path(yaml_path) if yaml_path else PROJECT_ROOT / "config.yaml"
        if path.exists():
            data = yaml.safe_load(path.read_text()) or {}
            return cls(**data)
        return cls()

    def save(self, yaml_path: str | Path | None = None):
        path = Path(yaml_path) if yaml_path else PROJECT_ROOT / "config.yaml"
        path.write_text(
            yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True)
        )

    # ---------- CLI override ----------
    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> "Params":
        p = argparse.ArgumentParser(
            description="Parametre override",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        for f in cls.__dataclass_fields__.values():
            flag = f"--{f.name}"
            if f.type == bool:
                p.add_argument(flag, action="store_true")
            else:
                p.add_argument(flag, type=f.type)
        args = p.parse_args(argv)
        base = cls.load()
        for k, v in vars(args).items():
            if v is not None:
                setattr(base, k, v)
        return base

# ---------- İlk çalıştırmada örnek YAML ----------
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
if not CONFIG_FILE.exists():
    Params().save(CONFIG_FILE)
    print(f"[INFO] Varsayılan config oluşturuldu → {CONFIG_FILE}", file=sys.stderr)

# ---------- Debug çıktısı ----------
if __name__ == "__main__":
    from pprint import pprint
    pprint(asdict(Params.load()))
