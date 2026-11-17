#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import yaml

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

from uav_continuum_whole_body_v2.environment import BoxWorld  # noqa: E402


def load_manifest(path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve(path, base):
    value = Path(path)
    return value if value.is_absolute() else (base / value).resolve()


def box_link(item, index):
    center = " ".join(str(value) for value in item["center"])
    size = " ".join(str(value) for value in item["size"])
    name = str(item.get("name", "box_%d" % index))
    return """    <model name="{name}">
      <static>true</static><pose>{center} 0 0 0</pose>
      <link name="link"><collision name="collision"><geometry><box><size>{size}</size></box></geometry></collision>
      <visual name="visual"><geometry><box><size>{size}</size></box></geometry>
      <material><ambient>0.55 0.36 0.18 1</ambient><diffuse>0.70 0.48 0.24 1</diffuse></material></visual></link>
    </model>""".format(name=name, center=center, size=size)


def generate(manifest_path, output_dir, build_esdf=False):
    manifest = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = resolve(manifest["base_world_boxes_file"], manifest_path.parent)
    with base_path.open("r", encoding="utf-8") as handle:
        base_boxes = (yaml.safe_load(handle) or {}).get("boxes", [])
    task_boxes = list(manifest.get("task_boxes", []))
    keepout = list(manifest.get("body_keepout_boxes", []))
    combined = base_boxes + task_boxes
    boxes_path = output_dir / "scenario_3c_unified_boxes.yaml"
    with boxes_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"boxes": combined, "body_keepout_boxes": keepout}, handle, sort_keys=False)
    sdf = "<?xml version=\"1.0\"?>\n<sdf version=\"1.6\">\n  <world name=\"scenario_3c_overlay\">\n"
    sdf += "\n".join(box_link(item, index) for index, item in enumerate(task_boxes))
    sdf += "\n  </world>\n</sdf>\n"
    sdf_path = output_dir / "scenario_3c_unified_overlay.sdf"
    sdf_path.write_text(sdf, encoding="utf-8")
    result = {"manifest": str(manifest_path), "base_boxes": len(base_boxes), "task_boxes": len(task_boxes),
              "combined_boxes": len(combined), "boxes": str(boxes_path), "sdf": str(sdf_path)}
    if build_esdf:
        bounds = manifest["map_bounds"]
        esdf = BoxWorld.from_dicts(combined).build_esdf(bounds["minimum"], bounds["maximum"], float(bounds["resolution"]))
        esdf_path = output_dir / "scenario_3c_unified_esdf.npz"
        esdf.save(esdf_path, {"manifest": str(manifest_path), "resolution": bounds["resolution"]})
        result["esdf"] = str(esdf_path)
    report = output_dir / "generation_report.json"
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate all scenario_3c planning/evaluation/overlay assets from one manifest")
    parser.add_argument("--manifest", type=Path, default=PACKAGE / "config" / "scenario_3c_unified.yaml")
    parser.add_argument("--output", type=Path, default=PACKAGE / "generated")
    parser.add_argument("--build-esdf", action="store_true", help="also generate the (slower) dense ESDF npz")
    args = parser.parse_args()
    print(json.dumps(generate(args.manifest.resolve(), args.output.resolve(), args.build_esdf), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
