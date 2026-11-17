"""Deterministic parameterized scenario families for comparison experiments."""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import yaml

from .geometry import world_arm_points
from .mapping import MapSnapshot, observed_box_points
from .types import ArmModel


SCENARIO_FAMILIES = ("passage_sweep", "multi_passage", "clutter_reach")
DIFFICULTIES = ("easy", "medium", "hard")


@dataclass
class ScenarioSpec:
    name: str
    family: str
    difficulty: str
    seed: int
    boxes: list
    bounds_minimum: list
    bounds_maximum: list
    start_state: list
    terminal_state: list
    target: list
    metadata: dict = field(default_factory=dict)

    @property
    def task(self):
        terminal = self.terminal_state
        # Passage tasks only require safe transit to the terminal region;
        # Clutter-Reach is an end-effector task and therefore uses the
        # manipulation-grade threshold defined by the experiment protocol.
        tip_tolerance = 0.03 if self.family == "clutter_reach" else 0.08
        return {
            "schema_version": 1,
            "frame_id": "world",
            "name": self.name,
            "family": self.family,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "start_state": self.start_state,
            "target": self.target + [0.0],
            "preferred_standoff": terminal[:4],
            "terminal_shape_deg": np.rad2deg(terminal[4:6]).tolist(),
            "tip_tolerance_m": tip_tolerance,
            "task_time_limit": 600.0,
        }

    @property
    def oracle(self):
        return {
            "schema_version": 1,
            "frame_id": "world",
            "name": self.name,
            "boxes": self.boxes,
            "oracle_only": True,
        }


def _box(name, center, size, yaw_deg=0.0):
    box = {
        "name": str(name),
        "center": [float(value) for value in center],
        "size": [float(value) for value in size],
    }
    if abs(float(yaw_deg)) > 1e-12:
        box["yaw_deg"] = float(yaw_deg)
    return box


def _passage_wall(name, y, lower_top, upper_bottom, width=6.0,
                  thickness=0.22, yaw_deg=0.0):
    lower_height = lower_top
    upper_height = 3.0 - upper_bottom
    return [
        _box(name + "_lower", [0.0, y, 0.5 * lower_height],
             [width, thickness, lower_height], yaw_deg=yaw_deg),
        _box(name + "_upper", [0.0, y, upper_bottom + 0.5 * upper_height],
             [width, thickness, upper_height], yaw_deg=yaw_deg),
    ]


def generate_scenario(family, difficulty="medium", seed=0):
    family = str(family).strip().lower()
    difficulty = str(difficulty).strip().lower()
    if family not in SCENARIO_FAMILIES:
        raise ValueError("unknown family %r" % family)
    if difficulty not in DIFFICULTIES:
        raise ValueError("unknown difficulty %r" % difficulty)
    level = DIFFICULTIES.index(difficulty)
    rng = np.random.RandomState(int(seed))
    boxes = []
    if family == "passage_sweep":
        gap = [1.06, 0.92, 0.84][level]
        center = 1.31 + rng.uniform(-0.03, 0.03)
        yaw_deg = [0.0, 8.0, 16.0][level] + rng.uniform(-2.0, 2.0)
        boxes.extend(_passage_wall(
            "passage", -3.0, center - 0.5 * gap, center + 0.5 * gap,
            thickness=0.18 + 0.04 * level, yaw_deg=yaw_deg,
        ))
        terminal_y = -5.5
    elif family == "multi_passage":
        base_gap = [1.08, 0.94, 0.86][level]
        angle_scale = [0.35, 0.70, 1.0][level]
        for index, y in enumerate((-2.4, -4.8, -7.2)):
            center = 1.30 + rng.uniform(-0.08, 0.08)
            gap = base_gap + rng.uniform(-0.025, 0.025)
            boxes.extend(_passage_wall(
                "passage_%d" % (index + 1), y,
                center - 0.5 * gap, center + 0.5 * gap,
                thickness=0.18 + 0.03 * level,
                yaw_deg=angle_scale * (-14.0, 9.0, -18.0)[index]
                + rng.uniform(-2.0, 2.0),
            ))
        terminal_y = -9.3
    else:
        gap = [1.10, 0.96, 0.88][level]
        boxes.extend(_passage_wall(
            "reach_gate", -4.0, 1.30 - 0.5 * gap, 1.30 + 0.5 * gap,
            thickness=0.20,
            yaw_deg=[0.0, 7.0, 13.0][level] + rng.uniform(-1.5, 1.5),
        ))
        cube_count = [4, 7, 10][level]
        for index in range(cube_count):
            y = rng.uniform(-7.5, -4.8)
            x = rng.choice([-1.0, 1.0]) * rng.uniform(0.65, 1.35)
            z = rng.uniform(0.65, 1.75)
            size = rng.uniform([0.25, 0.25, 0.25], [0.55, 0.55, 0.65])
            boxes.append(_box("clutter_%02d" % index, [x, y, z], size))
        terminal_y = -8.5

    start = np.asarray([0.0, 0.0, 1.30, -0.5 * np.pi, 0.0, 0.0])
    terminal = np.asarray([
        0.0, terminal_y, 1.30, -0.5 * np.pi,
        np.deg2rad(90.0), -np.pi,
    ])
    target = world_arm_points(terminal, ArmModel())[-1]
    name = "%s_%s_s%03d" % (family, difficulty, int(seed))
    return ScenarioSpec(
        name=name, family=family, difficulty=difficulty, seed=int(seed),
        boxes=boxes,
        bounds_minimum=[-2.5, terminal_y - 1.0, 0.0],
        bounds_maximum=[2.5, 1.0, 2.6],
        start_state=start.tolist(), terminal_state=terminal.tolist(),
        target=target.tolist(),
        metadata={
            "generator": "parameterized_v3_oriented_walls",
            "design_map_resolution_m": 0.05,
            "difficulty_rule": "gap_schedule_frozen_before_pilot",
        },
    )


def box_surface_points(box, resolution):
    center = np.asarray(box["center"], dtype=float)
    half = 0.5 * np.asarray(box["size"], dtype=float)
    axes = [
        np.linspace(-half[i], half[i], int(np.ceil(2.0 * half[i] / resolution)) + 1)
        for i in range(3)
    ]
    surfaces = []
    for fixed_axis in range(3):
        free = [axis for axis in range(3) if axis != fixed_axis]
        mesh = np.meshgrid(axes[free[0]], axes[free[1]], indexing="ij")
        for side in (-1.0, 1.0):
            points = np.empty((mesh[0].size, 3), dtype=float)
            points[:, fixed_axis] = side * half[fixed_axis]
            points[:, free[0]] = mesh[0].ravel()
            points[:, free[1]] = mesh[1].ravel()
            surfaces.append(points)
    points = np.vstack(surfaces)
    yaw = float(box.get("yaw", np.deg2rad(box.get("yaw_deg", 0.0))))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotated = points.copy()
    rotated[:, 0] = cosine * points[:, 0] - sine * points[:, 1]
    rotated[:, 1] = sine * points[:, 0] + cosine * points[:, 1]
    return rotated + center


def synthetic_development_snapshot(spec, resolution=0.10):
    occupied = np.vstack([
        box_surface_points(box, resolution) for box in spec.boxes
    ]) if spec.boxes else np.empty((0, 3))
    observed = observed_box_points(
        spec.bounds_minimum, spec.bounds_maximum, resolution
    )
    return MapSnapshot.from_points(
        occupied, observed, resolution=resolution,
        frame_id="world", source="synthetic_perfect_map",
        metadata={
            "scenario": spec.name,
            "truth_derived": True,
            "formal_experiment_allowed": False,
            "localization_source": "synthetic_fixture",
            "unknown_is_occupied": True,
        },
    )


def write_scenario(spec, output, development_snapshot=True, resolution=0.10):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "task.yaml").write_text(
        yaml.safe_dump(spec.task, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (output / "oracle_truth.yaml").write_text(
        yaml.safe_dump(spec.oracle, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    manifest = {
        "name": spec.name,
        "family": spec.family,
        "difficulty": spec.difficulty,
        "seed": spec.seed,
        "bounds": {"minimum": spec.bounds_minimum, "maximum": spec.bounds_maximum},
        "metadata": spec.metadata,
    }
    (output / "scenario_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if development_snapshot:
        synthetic_development_snapshot(spec, resolution=resolution).save(
            output / "development_map.npz"
        )
    return output
