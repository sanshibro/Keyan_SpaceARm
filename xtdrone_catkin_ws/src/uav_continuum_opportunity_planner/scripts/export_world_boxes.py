#!/usr/bin/env python3
import argparse
import math
import os
import xml.etree.ElementTree as ET

import yaml
import tf.transformations as tft


def pose_values(elem):
    if elem is None or not elem.text:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vals = [float(v) for v in elem.text.split()]
    return (vals + [0.0] * 6)[:6]


def pose_matrix(vals):
    mat = tft.euler_matrix(vals[3], vals[4], vals[5])
    mat[0, 3], mat[1, 3], mat[2, 3] = vals[0], vals[1], vals[2]
    return mat


def box_aabb(world_mat, size):
    hx, hy, hz = [0.5 * float(v) for v in size]
    corners = []
    for sx in [-hx, hx]:
        for sy in [-hy, hy]:
            for sz in [-hz, hz]:
                p = world_mat.dot([sx, sy, sz, 1.0])
                corners.append((p[0], p[1], p[2]))
    mins = [min(p[i] for p in corners) for i in range(3)]
    maxs = [max(p[i] for p in corners) for i in range(3)]
    center = [(mins[i] + maxs[i]) * 0.5 for i in range(3)]
    aabb_size = [maxs[i] - mins[i] for i in range(3)]
    return center, aabb_size


def inflate_degenerate_size(size, min_dim):
    return [max(abs(float(v)), min_dim) for v in size]


def should_include(model_name, collision_name, center, size, args):
    if any(model_name.startswith(prefix) for prefix in args.exclude_model_prefix):
        return False
    if args.include_model and model_name not in args.include_model:
        return False
    x, y, z = center
    if not (args.min_x <= x <= args.max_x and args.min_y <= y <= args.max_y and args.min_z <= z <= args.max_z):
        return False
    return True


def export_boxes(args):
    root = ET.parse(args.world).getroot()
    boxes = []
    for model in root.findall("./world/model"):
        model_name = model.get("name", "model")
        model_mat = pose_matrix(pose_values(model.find("pose")))
        for link in model.findall(".//link"):
            link_name = link.get("name", "link")
            link_mat = model_mat.dot(pose_matrix(pose_values(link.find("pose"))))
            for collision in link.findall("collision"):
                size_elem = collision.find("./geometry/box/size")
                if size_elem is None or not size_elem.text:
                    continue
                collision_name = collision.get("name", "collision")
                collision_mat = link_mat.dot(pose_matrix(pose_values(collision.find("pose"))))
                raw_size = inflate_degenerate_size(size_elem.text.split()[:3], args.min_box_dim)
                center, size = box_aabb(collision_mat, raw_size)
                if not should_include(model_name, collision_name, center, size, args):
                    continue
                boxes.append(
                    {
                        "name": "%s_%s_%s" % (args.name_prefix, model_name, collision_name),
                        "center": [float(round(float(v), 4)) for v in center],
                        "size": [float(round(max(float(v), 0.001), 4)) for v in size],
                    }
                )
    return boxes


def main():
    parser = argparse.ArgumentParser(description="Export SDF collision boxes as planner AABB YAML.")
    parser.add_argument("--world", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name-prefix", default="sdf")
    parser.add_argument("--include-model", action="append", default=[])
    parser.add_argument("--exclude-model-prefix", action="append", default=["ground_plane"])
    parser.add_argument("--min-box-dim", type=float, default=0.05)
    parser.add_argument("--min-x", type=float, default=-20.0)
    parser.add_argument("--max-x", type=float, default=20.0)
    parser.add_argument("--min-y", type=float, default=-20.0)
    parser.add_argument("--max-y", type=float, default=20.0)
    parser.add_argument("--min-z", type=float, default=0.02)
    parser.add_argument("--max-z", type=float, default=5.5)
    args = parser.parse_args()

    boxes = export_boxes(args)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        yaml.safe_dump({"boxes": boxes}, fh, default_flow_style=False, sort_keys=False)
    print("wrote %d boxes to %s" % (len(boxes), args.output))


if __name__ == "__main__":
    main()
