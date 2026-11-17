"""Geometry constants from the archived, unmodified Qingyu SDK configuration."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math
import yaml

VENDOR = Path(__file__).resolve().parent / 'vendor/qingyu_continuum_arm'

@lru_cache(maxsize=1)
def sdk_geometry():
    relative = 'config/continuum_arm_visualize.yaml'
    raw = (VENDOR / relative).read_bytes()
    provenance = json.loads((VENDOR / 'PROVENANCE.json').read_text())
    if hashlib.sha256(raw).hexdigest() != provenance['files'][relative]:
        raise ValueError('Archived Qingyu geometry configuration hash mismatch')
    config = yaml.safe_load(raw)
    for key in ('arc_length', 'visual_base_offset_z'):
        value = float(config[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid Qingyu geometry: ' + key)
    return float(config['arc_length']), float(config['visual_base_offset_z'])

ARC_LENGTH, BASE_LENGTH = sdk_geometry()
