#!/usr/bin/env python3
"""Generate publication figures from the frozen 90-pair/450-trial study.

Every plotted quantity is loaded from the final paired table or the measured
closed-loop logs.  The representative trajectory uses the recorded PX4 pose
and continuum shape; Oracle boxes are used only for post-hoc visualization and
clearance evaluation, exactly as in the experiment audit.
"""

import csv
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import yaml


SCRIPT = Path(__file__).resolve()
FINAL = SCRIPT.parent.parent
OUT = SCRIPT.parent
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
SOURCE_DATA = OUT / "source_data"
for directory in (FIGURES, TABLES, SOURCE_DATA):
    directory.mkdir(parents=True, exist_ok=True)


def repository_root(path):
    for candidate in (path,) + tuple(path.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("repository root not found")


REPO = repository_root(SCRIPT)
PACKAGE = REPO / "xtdrone_catkin_ws/src/uav_continuum_whole_body_v2"
sys.path.insert(0, str(PACKAGE / "src"))

from uav_continuum_whole_body_v2.environment import BoxWorld  # noqa: E402
from uav_continuum_whole_body_v2.geometry import (  # noqa: E402
    coupling_metrics_batch,
    world_arm_points_batch,
    world_arm_sections_batch,
    world_body_points_batch,
)
from uav_continuum_whole_body_v2.mapping import voxel_centers  # noqa: E402
from uav_continuum_whole_body_v2.types import (  # noqa: E402
    ArmModel,
    Limits,
    MassModel,
)


METHODS = (
    "fixed_morphology",
    "aet_sequential_minjerk",
    "informed_rrtstar_wb",
    "global_batch_wb",
    "rolling_wb_v2",
)
LABELS = {
    "fixed_morphology": "Fixed",
    "aet_sequential_minjerk": "AET-Seq.",
    "informed_rrtstar_wb": "WB-RRT*",
    "global_batch_wb": "Global-WB",
    "rolling_wb_v2": "Rolling-WB-V2",
}
COLORS = {
    "fixed_morphology": "#6F6F6F",
    "aet_sequential_minjerk": "#E69F00",
    "informed_rrtstar_wb": "#009E73",
    "global_batch_wb": "#56B4E9",
    "rolling_wb_v2": "#D55E00",
}
DIFFICULTIES = ("easy", "medium", "hard")
FAMILIES = ("passage_sweep", "multi_passage", "clutter_reach")
FAMILY_LABELS = {
    "passage_sweep": "Passage-Sweep",
    "multi_passage": "Multi-Passage",
    "clutter_reach": "Clutter-Reach",
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.2,
        "legend.fontsize": 7.4,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.75,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.28,
        "lines.linewidth": 1.35,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax, text, x=0.015, y=0.985):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9.2,
            fontweight="bold", va="top", ha="left", zorder=20,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78,
                  "pad": 0.3})


def save_figure(fig, stem):
    fig.savefig(str(FIGURES / (stem + ".pdf")), dpi=600)
    fig.savefig(str(FIGURES / (stem + ".png")), dpi=600)
    plt.close(fig)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_trials():
    with (FINAL / "paired_trials.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["success"] = row["success"] == "True"
        row["random_seed"] = int(row["random_seed"])
        for key in (
            "planning_time_s", "task_time_s", "deadline_miss_count",
            "max_hold_s", "nonterminal_hold_s", "body_clearance_m",
            "arm_clearance_m", "tracking_rmse_m", "com_offset_max_m",
            "jerk_energy", "tip_error_m",
        ):
            try:
                row[key] = float(row[key])
            except (TypeError, ValueError):
                row[key] = float("nan")
    return rows


def finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def method_values(rows, method, key, successes_only=False):
    return finite([
        row[key] for row in rows
        if row["method"] == method and (row["success"] or not successes_only)
    ])


def common_success_scenarios(rows):
    outcomes = {}
    for row in rows:
        outcomes.setdefault(row["scenario"], {})[row["method"]] = row["success"]
    return {
        scenario for scenario, methods in outcomes.items()
        if all(methods.get(method, False) for method in METHODS)
    }


def grouped_success(rows, group_key, groups):
    result = {}
    for group in groups:
        result[group] = {}
        for method in METHODS:
            selected = [row for row in rows
                        if row["method"] == method and row[group_key] == group]
            result[group][method] = (
                sum(row["success"] for row in selected) / float(len(selected))
            )
    return result


def make_overview(rows, summary, paired_stats):
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.6))
    x = np.arange(len(METHODS))

    ax = axes[0, 0]
    rates = np.asarray([summary[m]["success_rate"] for m in METHODS])
    low = np.asarray([summary[m]["success_ci95_low"] for m in METHODS])
    high = np.asarray([summary[m]["success_ci95_high"] for m in METHODS])
    bars = ax.bar(x, 100.0 * rates, color=[COLORS[m] for m in METHODS],
                  edgecolor="black", linewidth=0.45, width=0.72)
    ax.errorbar(x, 100.0 * rates,
                yerr=np.vstack((100.0 * (rates - low), 100.0 * (high - rates))),
                fmt="none", ecolor="black", elinewidth=0.8, capsize=2.4)
    for index, (bar, rate) in enumerate(zip(bars, rates)):
        ax.text(bar.get_x() + bar.get_width() / 2.0, 100.0 * rate + 2.2,
                "%d/90" % summary[METHODS[index]]["success_count"],
                ha="center", va="bottom", fontsize=6.9)
    ax.set_ylim(0, 111)
    ax.set_ylabel("Task success (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=18, ha="right")
    ax.grid(axis="y")
    ax.set_title("Overall success with Wilson 95% CI")
    panel_label(ax, "(a)")

    ax = axes[0, 1]
    difficulty = grouped_success(rows, "scenario_difficulty", DIFFICULTIES)
    width = 0.16
    gx = np.arange(len(DIFFICULTIES))
    for index, method in enumerate(METHODS):
        values = [100.0 * difficulty[group][method] for group in DIFFICULTIES]
        ax.bar(gx + (index - 2) * width, values, width=width,
               color=COLORS[method], label=LABELS[method],
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(gx)
    ax.set_xticklabels([value.title() for value in DIFFICULTIES])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Task success (%)")
    ax.set_title("Success versus scenario difficulty")
    ax.grid(axis="y")
    panel_label(ax, "(b)")

    ax = axes[1, 0]
    family = grouped_success(rows, "scenario_family", FAMILIES)
    gx = np.arange(len(FAMILIES))
    for index, method in enumerate(METHODS):
        values = [100.0 * family[group][method] for group in FAMILIES]
        ax.bar(gx + (index - 2) * width, values, width=width,
               color=COLORS[method], label=LABELS[method],
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(gx)
    ax.set_xticklabels([FAMILY_LABELS[value] for value in FAMILIES],
                       rotation=10, ha="right")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Task success (%)")
    ax.set_title("Success across independent scenario families")
    ax.grid(axis="y")
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    failure_groups = (
        ("No path / gate", {"NO_PATH", "PLANNING_OR_LIVE_GATE_FAILURE",
                            "repeated_planning_rejection"}, "#CC79A7"),
        ("Plan timeout", {"PLAN_TIMEOUT"}, "#F0E442"),
        ("Validation reject", {"VALIDATION_REJECT"}, "#0072B2"),
        ("Execution / task", {"TRACKING_ABORT", "TASK_TIMEOUT",
                              "EXECUTION_HANDOFF_ABORT"}, "#D55E00"),
    )
    bottom = np.zeros(len(METHODS))
    for label, categories, color in failure_groups:
        values = np.asarray([
            sum((not row["success"]) and row["failure_category"] in categories
                for row in rows if row["method"] == method)
            for method in METHODS
        ], dtype=float)
        ax.bar(x, values, bottom=bottom, color=color, label=label,
               edgecolor="white", linewidth=0.4)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=18, ha="right")
    ax.set_ylabel("Failed trials (count)")
    ax.set_ylim(0, max(bottom) * 1.20)
    for index, total in enumerate(bottom):
        ax.text(index, total + 0.35, "%d" % int(total), ha="center",
                va="bottom", fontsize=6.8, fontweight="bold")
    ax.set_title("Retained failure outcomes")
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=6.5)
    panel_label(ax, "(d)")

    handles = [Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in METHODS]
    fig.legend(handles, [LABELS[m] for m in METHODS], ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.005), frameon=False)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.95, bottom=0.14,
                        wspace=0.28, hspace=0.43)
    save_figure(fig, "fig01_success_and_failure_overview")


def colored_boxplot(ax, data):
    plot = ax.boxplot(data, patch_artist=True, widths=0.63, whis=[5, 95],
                      showfliers=False, medianprops={"color": "black", "linewidth": 1.2},
                      whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8},
                      boxprops={"linewidth": 0.8})
    for patch, method in zip(plot["boxes"], METHODS):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.78)
    ax.set_xticks(np.arange(1, len(METHODS) + 1))
    ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=20, ha="right")
    ax.grid(axis="y")


def make_continuous_metrics(rows):
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 5.25))
    common = common_success_scenarios(rows)
    specs = (
        ("planning_time_s", False, "Planning latency per trial (s)", False, None),
        ("task_time_s", True, "Task completion time (s)", False, None),
        ("body_clearance_m", True, "Oracle body clearance (m)", False, 0.06),
        ("arm_clearance_m", True, "Oracle arm clearance (m)", False, 0.02),
        ("tracking_rmse_m", True, "Tracking RMSE (m)", False, None),
        ("jerk_energy", True, "Executed jerk energy / m", True, None),
    )
    for index, (ax, spec) in enumerate(zip(axes.ravel(), specs)):
        key, _, ylabel, log_scale, threshold = spec
        data = [finite([row[key] for row in rows
                        if row["method"] == method and row["scenario"] in common])
                for method in METHODS]
        colored_boxplot(ax, data)
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale("log")
        if threshold is not None:
            ax.axhline(threshold, color="#B22222", linestyle="--", linewidth=1.0,
                       label="Acceptance threshold")
            ax.legend(frameon=False, loc="best")
        panel_label(ax, "(%s)" % chr(ord("a") + index),
                    y=0.985)
    fig.suptitle("Execution quality on %d commonly successful paired scenarios" %
                 len(common), fontsize=9.3, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.94, bottom=0.13,
                        wspace=0.38, hspace=0.46)
    save_figure(fig, "fig02_execution_quality_distributions")


def make_paired_matrix(rows):
    ordered = sorted(
        {row["scenario"] for row in rows},
        key=lambda scenario: (
            FAMILIES.index(next(row["scenario_family"] for row in rows
                                if row["scenario"] == scenario)),
            DIFFICULTIES.index(next(row["scenario_difficulty"] for row in rows
                                    if row["scenario"] == scenario)),
            next(row["random_seed"] for row in rows if row["scenario"] == scenario),
        ),
    )
    matrix = np.zeros((len(ordered), len(METHODS)), dtype=int)
    lookup = {(row["scenario"], row["method"]): row for row in rows}
    for i, scenario in enumerate(ordered):
        for j, method in enumerate(METHODS):
            matrix[i, j] = 1 if lookup[(scenario, method)]["success"] else 0

    fig, ax = plt.subplots(figsize=(5.05, 7.2))
    cmap = mpl_colors.ListedColormap(["#D9D9D9", "#0072B2"])
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(METHODS)))
    counts = [int(matrix[:, j].sum()) for j in range(len(METHODS))]
    ax.set_xticklabels(["%s\n%d/90" % (LABELS[m], count)
                        for m, count in zip(METHODS, counts)],
                       rotation=22, ha="right")
    ax.set_yticks([])
    ax.set_title("Paired closed-loop outcomes (90 scenarios)")
    for boundary in range(10, 90, 10):
        ax.axhline(boundary - 0.5, color="white", linewidth=0.8)
    for boundary in (30, 60):
        ax.axhline(boundary - 0.5, color="black", linewidth=1.25)
    for family_index, family in enumerate(FAMILIES):
        family_start = family_index * 30
        for difficulty_index, difficulty in enumerate(DIFFICULTIES):
            center = family_start + difficulty_index * 10 + 4.5
            label = "%s / %s" % (FAMILY_LABELS[family], difficulty.title())
            ax.text(-0.72, center, label, va="center", ha="right", fontsize=7.2,
                    clip_on=False)
    legend = [Rectangle((0, 0), 1, 1, color="#0072B2"),
              Rectangle((0, 0), 1, 1, color="#D9D9D9")]
    ax.legend(legend, ["Success", "Failure"], frameon=False, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, -0.16))
    fig.subplots_adjust(left=0.35, right=0.97, top=0.94, bottom=0.145)
    save_figure(fig, "fig03_paired_outcome_matrix")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def quaternion_yaw(quaternion):
    x, y, z, w = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def representative_paths():
    scenario = "multi_passage_hard_s009"
    scenario_root = (
        REPO / "xtdrone_plan/experiments/method_comparison/"
        "schema6_formal_expansion_commit4f80214/wave23_hard_seed9/"
        "multi_passage_hard_s009__deebe99f9dbd"
    )
    candidates = sorted((scenario_root / "closed_loop_runs").glob(
        scenario + "__rolling_wb_v2__*"
    ))
    if len(candidates) != 1:
        raise RuntimeError("representative V2 run is not unique")
    run = candidates[0]
    oracle = (
        REPO / "xtdrone_plan/experiments/method_comparison/"
        "formal_schema5_main_90_commitadfdcdb/scenarios/" / scenario / "oracle_truth.yaml"
    )
    task = oracle.with_name("task.yaml")
    map_file = scenario_root / "audited_frozen_map.npz"
    return run, oracle, task, map_file


def representative_data():
    run, oracle_path, task_path, map_path = representative_paths()
    acceptance = load_json(run / "acceptance_report.json")
    events = read_jsonl(run / "events.jsonl")
    tracking = read_jsonl(run / "tracking.jsonl")
    oracle = yaml.safe_load(oracle_path.read_text(encoding="utf-8"))
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))

    start_candidates = [row.get("actual_start") for row in events
                        if row.get("state") == "START_SCHEDULED" and row.get("actual_start")]
    task_complete = [row["stamp"] for row in events if row.get("state") == "TASK_COMPLETE"]
    start = min(start_candidates)
    finish = max(task_complete)
    measured = [row for row in tracking if start <= row["stamp"] <= finish]
    stamps = np.asarray([row["stamp"] for row in measured], dtype=float)
    positions = np.asarray([row["position"] for row in measured], dtype=float)
    yaw = np.unwrap(np.asarray([quaternion_yaw(row["orientation"]) for row in measured]))
    arm_deg = np.asarray([row["arm_deg"] for row in measured], dtype=float)
    states = np.column_stack((positions, yaw, np.deg2rad(arm_deg)))

    switches = {}
    for row in events:
        if row.get("state") == "SWITCHED_ROLLING" and row.get("trajectory_id"):
            switches[int(row["trajectory_id"])] = float(row["actual_start"])
    switch_times = [start] + [switches[key] for key in sorted(switches)]
    planning = acceptance["all_planning_attempt_time_s"]["values"]
    switch_times = np.asarray(switch_times[:len(planning)], dtype=float)

    # Remaining duration of the previously certified trajectory when a new
    # candidate finishes dense validation. This is reconstructed only from
    # frozen trajectory metadata and SAFE_TRAJECTORY event stamps.
    metadata = {}
    for path in sorted(run.glob("trajectory_*.json")):
        item = load_json(path)
        metadata[int(item["trajectory_id"])] = item
    safe_events = {}
    for row in events:
        if row.get("state") == "SAFE_TRAJECTORY" and row.get("trajectory_id"):
            safe_events[int(row["trajectory_id"])] = float(row["stamp"])
    safe_remaining = []
    for identifier in sorted(metadata)[:len(planning)]:
        if identifier <= 1 or identifier - 1 not in metadata:
            safe_remaining.append(float("nan"))
            continue
        previous = metadata[identifier - 1]
        previous_end = (float(previous["execution_start_time"])
                        + float(previous["duration"]))
        safe_remaining.append(previous_end - safe_events[identifier])

    resolution_and_map = np.load(str(map_path), allow_pickle=False)
    resolution = float(resolution_and_map["resolution"][0])
    occupied = voxel_centers(resolution_and_map["occupied_indices"], resolution)

    world = BoxWorld.from_dicts(oracle["boxes"])
    arm = ArmModel()
    limits = Limits(body_radius=0.28, body_height=0.25, body_margin=0.10,
                    arm_margin=0.07, com_max=0.08, velocity_max=0.25,
                    acceleration_max=1.0, jerk_max=3.0,
                    yaw_rate_max=np.deg2rad(45.0),
                    yaw_acceleration_max=np.deg2rad(90.0),
                    tilt_max=np.deg2rad(15.0), thrust_min_ratio=0.5,
                    thrust_max_ratio=1.6, theta_rate_max=np.deg2rad(50.0),
                    phi_rate_max=np.deg2rad(60.0),
                    theta_acceleration_max=np.deg2rad(100.0),
                    phi_acceleration_max=np.deg2rad(120.0), z_min=0.75, z_max=2.2)
    mass = MassModel(1.50, 0.30, 0.25, 0.20)

    arm_points = world_arm_points_batch(states, arm)
    _, flexible, tool = world_arm_sections_batch(states, arm)
    body_points = world_body_points_batch(states, limits)
    body_clearance, _ = world.minimum_distances_grouped(body_points)
    arm_clearance, _ = world.minimum_distances_grouped(arm_points)
    arm_clearance = arm_clearance - arm.radius
    com_offset = coupling_metrics_batch(states, arm, mass)

    return {
        "run": run, "acceptance": acceptance, "events": events,
        "oracle": oracle, "task": task, "start": start, "finish": finish,
        "stamps": stamps, "time": stamps - start, "positions": positions,
        "arm_deg": arm_deg, "states": states, "flexible": flexible,
        "tool": tool, "tip": tool[:, -1, :], "occupied": occupied,
        "world": world, "arm": arm, "limits": limits,
        "body_clearance": body_clearance, "arm_clearance": arm_clearance,
        "com_offset": com_offset, "switch_times": switch_times - start,
        "planning": np.asarray(planning, dtype=float),
        "safe_buffer_remaining": np.asarray(safe_remaining, dtype=float),
    }


def rotated_rectangle_xy(box):
    center = np.asarray(box.center[:2], dtype=float)
    half = np.asarray(box.size[:2], dtype=float) * 0.5
    local = np.asarray([[-half[0], -half[1]], [half[0], -half[1]],
                        [half[0], half[1]], [-half[0], half[1]]])
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    rotation = np.asarray([[c, -s], [s, c]])
    return local.dot(rotation.T) + center


def ghost_indices(positions, count=11):
    distance = np.concatenate(([0.0], np.cumsum(np.linalg.norm(
        np.diff(positions, axis=0), axis=1))))
    targets = np.linspace(0.02 * distance[-1], 0.98 * distance[-1], count)
    indices = [int(np.argmin(np.abs(distance - value))) for value in targets]
    return sorted(set(indices))


def straight_arm_gate_counterfactual(case):
    """Evaluate a straight arm at the measured UAV pose of each passage.

    Only morphology is changed: UAV position and yaw remain measured values.
    This makes the overlay a controlled counterfactual rather than an
    alternative flight path.
    """
    gate_y = sorted({float(box.center[1]) for box in case["world"].boxes},
                    reverse=True)
    indices = [int(np.argmin(np.abs(case["positions"][:, 1] - value)))
               for value in gate_y]
    straight_states = case["states"][indices].copy()
    straight_states[:, 4:6] = 0.0
    straight_points = world_arm_points_batch(straight_states, case["arm"])
    _, flexible, tool = world_arm_sections_batch(straight_states, case["arm"])
    clearance, _ = case["world"].minimum_distances_grouped(straight_points)
    clearance = clearance - case["arm"].radius
    return {
        "gate_y": np.asarray(gate_y),
        "indices": np.asarray(indices, dtype=int),
        "flexible": flexible,
        "tool": tool,
        "clearance": clearance,
        "actual_clearance": case["arm_clearance"][indices],
    }


def passage_vertical_feasibility(case, map_inflation=0.02):
    """Robust UAV-center bounds for a straight arm in each vertical gate."""
    paired = {}
    for box in case["world"].boxes:
        if box.name.endswith("_lower"):
            paired.setdefault(box.name[:-6], {})["lower"] = box
        elif box.name.endswith("_upper"):
            paired.setdefault(box.name[:-6], {})["upper"] = box
    zero = np.zeros((1, 6), dtype=float)
    straight_points = world_arm_points_batch(zero, case["arm"])[0]
    straight_drop = -float(np.min(straight_points[:, 2]))
    results = []
    for name, pair in sorted(paired.items(), key=lambda item: item[1]["lower"].center[1],
                             reverse=True):
        lower, upper = pair["lower"], pair["upper"]
        lower_top = float(lower.center[2] + lower.half[2])
        upper_bottom = float(upper.center[2] - upper.half[2])
        straight_min = (lower_top + map_inflation + straight_drop +
                        case["arm"].radius + case["limits"].arm_margin)
        body_max = (upper_bottom - map_inflation -
                    0.5 * case["limits"].body_height -
                    case["limits"].body_margin)
        results.append({
            "name": name, "y": float(lower.center[1]),
            "lower_top": lower_top, "upper_bottom": upper_bottom,
            "straight_uav_z_min": straight_min,
            "body_uav_z_max": body_max,
            "robust_slack": body_max - straight_min,
        })
    return results


def plot_obstacle_top(ax, world):
    for box in world.boxes:
        polygon = Polygon(rotated_rectangle_xy(box), closed=True,
                          facecolor="#BDBDBD", edgecolor="#3A3A3A",
                          linewidth=0.75, alpha=0.55)
        ax.add_patch(polygon)


def plot_obstacle_side(ax, world):
    for box in world.boxes:
        half = box.world_aabb_half
        ax.add_patch(Rectangle((box.center[1] - half[1], box.center[2] - half[2]),
                               2.0 * half[1], 2.0 * half[2],
                               facecolor="#BDBDBD", edgecolor="#3A3A3A",
                               linewidth=0.75, alpha=0.55))


def box_vertices(box, x_clip=1.55):
    half = np.asarray(box.size, dtype=float) * 0.5
    half[0] = min(half[0], x_clip)
    local = np.asarray([[sx * half[0], sy * half[1], sz * half[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    rotation = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    vertices = local.dot(rotation.T) + np.asarray(box.center)
    indices = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
               (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
    return [[vertices[index] for index in face] for face in indices]


def make_whole_body_trajectory(case):
    fig = plt.figure(figsize=(7.16, 5.15))
    ax_top = fig.add_subplot(2, 2, 1)
    ax_side = fig.add_subplot(2, 2, 3)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    positions, tip = case["positions"], case["tip"]
    indices = ghost_indices(positions)
    cmap = plt.get_cmap("viridis")
    norm = mpl_colors.Normalize(0.0, case["time"][-1])
    vertical_bounds = passage_vertical_feasibility(case)

    occupied = case["occupied"]
    sample_step = max(1, len(occupied) // 7000)
    sampled = occupied[::sample_step]

    ax_top.scatter(sampled[:, 0], sampled[:, 1], s=0.45, c="#A6A6A6",
                   alpha=0.18, rasterized=True, label="SLAM occupied voxels")
    plot_obstacle_top(ax_top, case["world"])
    ax_top.plot(positions[:, 0], positions[:, 1], color="#0072B2", lw=1.7,
                label="Closed-loop simulated UAV path")
    ax_top.plot(tip[:, 0], tip[:, 1], color="#D55E00", lw=1.2,
                label="Closed-loop simulated tool-tip path")
    for index in indices:
        color = cmap(norm(case["time"][index]))
        backbone = np.vstack((case["flexible"][index], case["tool"][index]))
        ax_top.plot(backbone[:, 0], backbone[:, 1], color=color, lw=1.45)
        ax_top.add_patch(Circle((positions[index, 0], positions[index, 1]),
                                case["limits"].body_radius, fill=False,
                                edgecolor=color, linewidth=0.65, alpha=0.8))
    ax_top.set_xlabel("World x (m)")
    ax_top.set_ylabel("World y (m)")
    ax_top.set_xlim(-1.65, 1.65)
    ax_top.set_ylim(-10.1, 0.6)
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.grid()
    ax_top.set_title("Closed-loop simulated whole-body motion (top)")
    panel_label(ax_top, "(a)")

    mask = (sampled[:, 0] > -1.5) & (sampled[:, 0] < 1.5)
    ax_side.scatter(sampled[mask, 1], sampled[mask, 2], s=0.45,
                    c="#A6A6A6", alpha=0.16, rasterized=True)
    plot_obstacle_side(ax_side, case["world"])
    ax_side.plot(positions[:, 1], positions[:, 2], color="#0072B2", lw=1.7)
    ax_side.plot(tip[:, 1], tip[:, 2], color="#D55E00", lw=1.2)
    counterfactual = straight_arm_gate_counterfactual(case)
    gate_y = [item["y"] for item in vertical_bounds]
    map_inflation = 0.02
    gate_indices = list(counterfactual["indices"])
    body_half_y = (case["limits"].body_radius + case["limits"].body_margin +
                   map_inflation)
    body_half_z = (0.5 * case["limits"].body_height +
                   case["limits"].body_margin + map_inflation)
    arm_envelope_radius = (case["arm"].radius + case["limits"].arm_margin +
                           map_inflation)

    # Four faint, measured transition snapshots retain the temporal story
    # without competing visually with the three gate counterfactuals.
    transition_targets = [0.5 * (positions[0, 1] + gate_y[0])]
    transition_targets.extend(0.5 * (left + right)
                              for left, right in zip(gate_y[:-1], gate_y[1:]))
    transition_targets.append(0.5 * (gate_y[-1] + positions[-1, 1]))
    transition_indices = [int(np.argmin(np.abs(positions[:, 1] - value)))
                          for value in transition_targets]
    for index in transition_indices:
        backbone = np.vstack((case["flexible"][index], case["tool"][index]))
        ax_side.plot(backbone[:, 1], backbone[:, 2], color="#6F9E91",
                     lw=1.25, alpha=0.62, zorder=6)
        ax_side.add_patch(Rectangle(
            (positions[index, 1] - case["limits"].body_radius,
             positions[index, 2] - 0.5 * case["limits"].body_height),
            2.0 * case["limits"].body_radius,
            case["limits"].body_height,
            fill=False, edgecolor="#6F9E91", linewidth=0.65,
            alpha=0.62, zorder=6))

    for gate_number, index in enumerate(gate_indices):
        gate = vertical_bounds[gate_number]
        ax_side.add_patch(Rectangle(
            (positions[index, 1] - body_half_y, positions[index, 2] - body_half_z),
            2.0 * body_half_y, 2.0 * body_half_z,
            facecolor="#56B4E9", edgecolor="#0072B2", linestyle=":",
            linewidth=0.65, alpha=0.13, zorder=4))
        actual_backbone = np.vstack((case["flexible"][index],
                                     case["tool"][index]))
        straight_backbone = np.vstack((counterfactual["flexible"][gate_number],
                                       counterfactual["tool"][gate_number]))
        for point in actual_backbone:
            ax_side.add_patch(Circle(
                (point[1], point[2]), arm_envelope_radius,
                facecolor="#009E73", edgecolor="none", alpha=0.028,
                zorder=5))
        for point in straight_backbone:
            ax_side.add_patch(Circle(
                (point[1], point[2]), arm_envelope_radius,
                facecolor="#D62728", edgecolor="none", alpha=0.018,
                zorder=5))
        # Actual measured morphology and a controlled counterfactual in which
        # only theta/phi are set to zero.  UAV position and yaw are unchanged.
        ax_side.plot(actual_backbone[:, 1], actual_backbone[:, 2],
                     color="#009E73", lw=2.05, zorder=8)
        ax_side.plot(straight_backbone[:, 1], straight_backbone[:, 2],
                     color="#D62728", lw=1.35, linestyle="--", zorder=7)
        ax_side.add_patch(Rectangle((positions[index, 1] - case["limits"].body_radius,
                                     positions[index, 2] - 0.5 * case["limits"].body_height),
                                    2.0 * case["limits"].body_radius,
                                    case["limits"].body_height,
                                    fill=False, edgecolor="#0072B2", linewidth=0.9,
                                    zorder=7))
        # The red cross marks the lower-obstacle intersection of the straight
        # physical arm.  The label reports Oracle clearance at the same pose.
        collision_z = gate["lower_top"] + 0.018
        ax_side.plot([gate["y"]], [collision_z], marker="x", markersize=5.0,
                     markeredgewidth=1.1, color="#D62728", zorder=10)
        ax_side.annotate(
            "straight {:+.1f} cm".format(
                100.0 * counterfactual["clearance"][gate_number]),
            xy=(gate["y"], collision_z), xytext=(0, -10),
            textcoords="offset points", ha="center", va="top",
            fontsize=5.2, color="#B22222", zorder=11,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.82})

        # Show why simply lifting the straight-arm vehicle is not a robust
        # alternative: at the height needed to clear the lower obstacle, the
        # inflated body envelope penetrates the inflated upper constraint.
        lifted_z = gate["straight_uav_z_min"]
        ax_side.add_patch(Rectangle(
            (gate["y"] - body_half_y, lifted_z - body_half_z),
            2.0 * body_half_y, 2.0 * body_half_z,
            fill=False, edgecolor="#D62728", linestyle=(0, (2.0, 1.5)),
            linewidth=0.75, alpha=0.75, zorder=6))
        ax_side.annotate(
            "no safe lift\nslack {:+.1f} cm".format(
                100.0 * gate["robust_slack"]),
            xy=(gate["y"] - 0.36, gate["upper_bottom"] - 0.015),
            xytext=(-2, 9), textcoords="offset points",
            ha="center", va="bottom", fontsize=4.9, color="#B22222",
            arrowprops={"arrowstyle": "-|>", "color": "#B22222",
                        "lw": 0.55}, zorder=11,
            bbox={"boxstyle": "round,pad=0.10", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.82})
    ax_side.set_xlabel("World y (m)")
    ax_side.set_ylabel("World z (m)")
    ax_side.set_xlim(0.6, -10.1)
    ax_side.set_ylim(0.15, 2.55)
    ax_side.grid()
    ax_side.set_title("Closed-loop simulated morphology and safety envelopes (side view)")
    safety_handles = [
        matplotlib.lines.Line2D([], [], color="#009E73", lw=2.0),
        matplotlib.lines.Line2D([], [], color="#6F9E91", lw=1.25,
                                alpha=0.62),
        matplotlib.lines.Line2D([], [], color="#D62728", lw=1.4,
                                linestyle="--"),
        Rectangle((0, 0), 1, 1, facecolor="#56B4E9", edgecolor="#0072B2",
                  alpha=0.18),
    ]
    ax_side.legend(safety_handles,
                   ["Gate morphology", "Simulated transition",
                    "Straight-arm counterfactual", "UAV safety envelope"],
                   frameon=True, framealpha=0.88, ncol=2, fontsize=5.1,
                   loc="upper center")
    panel_label(ax_side, "(b)", y=0.84)

    for box in case["world"].boxes:
        collection = Poly3DCollection(box_vertices(box), facecolor="#BDBDBD",
                                      edgecolor="#5A5A5A", linewidth=0.35,
                                      alpha=0.24)
        ax_3d.add_collection3d(collection)
    ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2],
               color="#0072B2", lw=1.7)
    ax_3d.plot(tip[:, 0], tip[:, 1], tip[:, 2], color="#D55E00", lw=1.2)
    for index in indices:
        color = cmap(norm(case["time"][index]))
        backbone = np.vstack((case["flexible"][index], case["tool"][index]))
        ax_3d.plot(backbone[:, 0], backbone[:, 1], backbone[:, 2],
                   color=color, lw=1.5)
        ax_3d.scatter([positions[index, 0]], [positions[index, 1]],
                      [positions[index, 2]], color=[color], s=9)
    ax_3d.set_xlim(-1.55, 1.55)
    ax_3d.set_ylim(-10.0, 0.5)
    ax_3d.set_zlim(0.2, 2.5)
    ax_3d.set_xlabel("x (m)", labelpad=1)
    ax_3d.set_ylabel("y (m)", labelpad=1)
    # The time colorbar already occupies the right margin; z ticks retain the
    # vertical scale without colliding with the colorbar label.
    ax_3d.set_zlabel("")
    ax_3d.view_init(elev=21, azim=-62)
    ax_3d.set_title("Closed-loop simulated path and arm snapshots (3-D)")
    ax_3d.text2D(0.015, 0.915, "(c)", transform=ax_3d.transAxes,
                 fontsize=9.5, fontweight="bold", va="top",
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78})

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=[ax_top, ax_side, ax_3d],
                            fraction=0.023, pad=0.055)
    colorbar.set_label("Elapsed execution time (s)")
    handles, labels = ax_top.get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.08, right=0.885, top=0.94, bottom=0.14,
                        wspace=0.28, hspace=0.34)
    save_figure(fig, "fig04_measured_whole_body_trajectory")


def load_planned_windows(run):
    windows = []
    for npz_path in sorted(run.glob("trajectory_*.npz")):
        metadata = load_json(npz_path.with_suffix(".json"))
        data = np.load(str(npz_path))
        state_columns = metadata.get("state_columns", [])
        if "theta_deg" not in state_columns:
            raise RuntimeError("trajectory metadata lacks theta_deg: %s" % npz_path)
        windows.append({
            "id": int(metadata["trajectory_id"]),
            "start": float(metadata["execution_start_time"]),
            "time": np.asarray(data["time"], dtype=float),
            "states": np.asarray(data["states"], dtype=float),
            "theta_index": state_columns.index("theta_deg"),
        })
    return windows


def make_rolling_horizons(case):
    windows = load_planned_windows(case["run"])
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.75))
    cmap = plt.get_cmap("plasma")
    colors = [cmap(value) for value in np.linspace(0.05, 0.9, len(windows))]

    ax = axes[0]
    plot_obstacle_top(ax, case["world"])
    for window, color in zip(windows, colors):
        state = window["states"]
        ax.plot(state[:, 0], state[:, 1], color=color, lw=1.0, alpha=0.84,
                label="W%d" % window["id"])
        ax.scatter(state[0, 0], state[0, 1], color=[color], s=8, zorder=4)
    ax.plot(case["positions"][:, 0], case["positions"][:, 1], color="black",
            lw=1.55, label="Executed")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-10.0, 0.45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World x (m)")
    ax.set_ylabel("World y (m)")
    ax.set_title("Accepted rolling windows")
    ax.grid()
    panel_label(ax, "(a)")

    ax = axes[1]
    plot_obstacle_side(ax, case["world"])
    for window, color in zip(windows, colors):
        state = window["states"]
        ax.plot(state[:, 1], state[:, 2], color=color, lw=1.0, alpha=0.84)
        ax.scatter(state[0, 1], state[0, 2], color=[color], s=8, zorder=4)
    ax.plot(case["positions"][:, 1], case["positions"][:, 2], color="black", lw=1.55)
    ax.set_xlim(0.45, -10.0)
    ax.set_ylim(0.2, 2.5)
    ax.set_xlabel("World y (m)")
    ax.set_ylabel("World z (m)")
    ax.set_title("Replanning in the vertical plane")
    ax.grid()
    panel_label(ax, "(b)")

    ax = axes[2]
    for window, color in zip(windows, colors):
        state = window["states"]
        progress = -state[:, 1]
        ax.plot(progress, state[:, window["theta_index"]], color=color, lw=1.1,
                label="W%d" % window["id"])
    ax.plot(-case["positions"][:, 1], case["arm_deg"][:, 0],
            color="black", lw=1.55, label="Executed")
    ax.set_xlabel("Forward progress, -y (m)")
    ax.set_ylabel(r"Bending angle $\theta$ (deg)")
    ax.set_ylim(-5, 123)
    ax.set_title("Updated morphology schedules")
    ax.grid()
    panel_label(ax, "(c)", y=0.69)
    handles = [Rectangle((0, 0), 1, 1, color=color) for color in colors]
    handles.append(matplotlib.lines.Line2D([], [], color="black", lw=1.55))
    labels = ["Window %d" % window["id"] for window in windows] + ["Executed"]
    fig.legend(handles, labels,
               ncol=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               fontsize=6.5)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.89, bottom=0.245, wspace=0.35)
    save_figure(fig, "fig05_rolling_horizon_updates")


def make_case_time_history(case):
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 6.2), sharex=False)
    time = case["time"]
    switch_times = case["switch_times"]

    def switches(ax):
        for value in switch_times[1:]:
            ax.axvline(value, color="#999999", lw=0.6, alpha=0.7, linestyle=":")

    ax = axes[0, 0]
    ax.plot(time, -case["positions"][:, 1], color="#0072B2", label="UAV")
    ax.plot(time, -case["tip"][:, 1], color="#D55E00", label="Tool tip")
    switches(ax)
    ax.set_ylabel("Forward progress, -y (m)")
    ax.set_title("Closed-loop simulated task progress")
    ax.legend(frameon=False)
    ax.grid()
    panel_label(ax, "(a)")

    ax = axes[0, 1]
    ax.plot(time, case["positions"][:, 2], color="#0072B2", label="UAV z")
    ax.plot(time, case["tip"][:, 2], color="#D55E00", label="Tool-tip z")
    ax.axhspan(0.75, 2.2, color="#56B4E9", alpha=0.07)
    switches(ax)
    ax.set_ylabel("Height (m)")
    ax.set_title("UAV and tool-tip height")
    ax.legend(frameon=False)
    ax.grid()
    panel_label(ax, "(b)")

    ax = axes[1, 0]
    theta = case["arm_deg"][:, 0]
    phi = np.deg2rad(case["arm_deg"][:, 1])
    bending_x = theta * np.cos(phi)
    bending_y = theta * np.sin(phi)
    ax.plot(time, theta, color="#009E73", label=r"$\theta$")
    ax.plot(time, bending_x, color="#CC79A7", label=r"$b_x=\theta\cos\phi$")
    ax.plot(time, bending_y, color="#E69F00", label=r"$b_y=\theta\sin\phi$")
    ax.axhline(120.0, color="#B22222", linestyle="--", lw=0.8)
    switches(ax)
    ax.set_ylabel("Bending magnitude/components (deg)")
    ax.set_title("Executed simulated morphology vector")
    ax.legend(frameon=False, loc="upper right")
    ax.grid()
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    ax.plot(time, case["body_clearance"], color="#0072B2", label="Body")
    ax.plot(time, case["arm_clearance"], color="#D55E00", label="Arm")
    ax.axhline(0.06, color="#0072B2", linestyle="--", lw=0.75)
    ax.axhline(0.02, color="#D55E00", linestyle="--", lw=0.75)
    switches(ax)
    ax.set_ylabel("Oracle clearance (m)")
    ax.set_ylim(0, 0.5)
    ax.set_title("Post-hoc swept-body clearance (critical range)")
    ax.legend(frameon=False)
    ax.text(0.985, 0.73, "min body = %.3f m\nmin arm = %.3f m" %
            (np.min(case["body_clearance"]), np.min(case["arm_clearance"])),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
            bbox={"facecolor": "white", "edgecolor": "#BDBDBD", "alpha": 0.86})
    ax.grid()
    panel_label(ax, "(d)")

    event_rows = [row for row in case["events"]
                  if row.get("tracking_error") is not None
                  and case["start"] <= row["stamp"] <= case["finish"]]
    event_time = np.asarray([row["stamp"] - case["start"] for row in event_rows])
    tracking_error = np.asarray([row["tracking_error"] for row in event_rows])
    ax = axes[2, 0]
    ax.plot(event_time, tracking_error, color="#0072B2", label="Tracking error")
    ax.axhline(0.25, color="#0072B2", linestyle="--", lw=0.75)
    ax2 = ax.twinx()
    ax2.plot(time, case["com_offset"], color="#E69F00", label="COM offset")
    ax2.axhline(0.08, color="#E69F00", linestyle="--", lw=0.75)
    switches(ax)
    ax.set_xlabel("Elapsed execution time (s)")
    ax.set_ylabel("Tracking error (m)", color="#0072B2")
    ax2.set_ylabel("COM offset (m)", color="#E69F00")
    ax.set_title("Execution and coupling constraints")
    lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper right")
    ax.grid()
    panel_label(ax, "(e)")

    ax = axes[2, 1]
    x = case["switch_times"]
    values = case["planning"]
    ax.bar(x, values, width=2.1, color="#D55E00", edgecolor="black", linewidth=0.4)
    for xi, value, index in zip(x, values, range(1, len(values) + 1)):
        ax.text(xi, value + 0.04, "W%d" % index, ha="center", va="bottom", fontsize=6.4)
    ax.set_xlabel("Elapsed execution time (s)")
    ax.set_ylabel("Planning + validation (s)")
    ax.axhline(15.0, color="#B22222", linestyle="--", lw=0.9,
               label="15 s deadline")
    buffer_values = case["safe_buffer_remaining"]
    valid_buffer = np.isfinite(buffer_values)
    ax.scatter(x[valid_buffer], buffer_values[valid_buffer], color="#0072B2",
               marker="D", s=13, label="Certified buffer remaining")
    upper = max(15.8, float(np.nanmax(buffer_values)) * 1.12)
    ax.set_ylim(0, upper)
    ax.set_title("Deadline and certified-buffer margin")
    ax.legend(frameon=False, loc="upper right", fontsize=5.8)
    ax.grid(axis="y")
    panel_label(ax, "(f)")

    fig.subplots_adjust(left=0.085, right=0.91, top=0.96, bottom=0.08,
                        wspace=0.42, hspace=0.45)
    save_figure(fig, "fig06_representative_execution_time_history")


def write_source_data(rows, case, summary, paired_stats):
    main_columns = [
        "method", "success", "trials", "success_rate", "ci95_low", "ci95_high",
        "task_time_median_s", "planning_median_s", "planning_p95_s",
        "deadline_miss_rate", "body_clearance_p05_m", "arm_clearance_p05_m",
        "tracking_rmse_median_m", "com_offset_max_p95_m", "jerk_energy_median",
        "tip_error_median_m",
    ]
    with (SOURCE_DATA / "main_results.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(main_columns)
        for method in METHODS:
            item = summary[method]
            writer.writerow([
                method, item["success_count"], item["trials"], item["success_rate"],
                item["success_ci95_low"], item["success_ci95_high"],
                item["task_time_median_s"], item["planning_median_s"],
                item["planning_p95_s"], item["deadline_miss_rate"],
                item["body_clearance_p05_m"], item["arm_clearance_p05_m"],
                item["tracking_rmse_median_m"], item["com_offset_max_p95_m"],
                item["jerk_energy_median"], item["tip_error_median_m"],
            ])

    with (SOURCE_DATA / "mcnemar_tests.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["baseline", "paired_trials", "ours_only_success",
                         "baseline_only_success", "exact_two_sided_p"])
        for method in METHODS[:-1]:
            item = paired_stats["success_mcnemar"][method]
            writer.writerow([
                method, item["paired_trials"], item["ours_only_success"],
                item["baseline_only_success"], item["exact_two_sided_p"],
            ])

    common = common_success_scenarios(rows)
    metric_columns = ["planning_time_s", "task_time_s", "body_clearance_m",
                      "arm_clearance_m", "tracking_rmse_m", "jerk_energy"]
    with (SOURCE_DATA / "common_success_execution_metrics.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "method"] + metric_columns)
        for row in rows:
            if row["scenario"] in common:
                writer.writerow([row["scenario"], row["method"]] +
                                [row[key] for key in metric_columns])

    with (SOURCE_DATA / "success_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group_type", "group", "method", "success", "trials", "rate"])
        for group_type, group_key, groups in (
            ("difficulty", "scenario_difficulty", DIFFICULTIES),
            ("family", "scenario_family", FAMILIES),
        ):
            for group in groups:
                for method in METHODS:
                    selected = [row for row in rows
                                if row[group_key] == group and row["method"] == method]
                    count = sum(row["success"] for row in selected)
                    writer.writerow([group_type, group, method, count, len(selected),
                                     count / float(len(selected))])

    columns = ["time_s", "x_m", "y_m", "z_m", "theta_deg", "phi_deg",
               "tip_x_m", "tip_y_m", "tip_z_m", "body_clearance_m",
               "arm_clearance_m", "com_offset_m"]
    with (SOURCE_DATA / "representative_measured_timeline.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index in range(len(case["time"])):
            writer.writerow([
                case["time"][index], *case["positions"][index],
                *case["arm_deg"][index], *case["tip"][index],
                case["body_clearance"][index], case["arm_clearance"][index],
                case["com_offset"][index],
            ])

    with (SOURCE_DATA / "rolling_window_latency.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trajectory_id", "execution_time_s",
                         "planning_validation_time_s",
                         "certified_buffer_remaining_after_validation_s"])
        for index, (time_value, latency, remaining) in enumerate(
                zip(case["switch_times"], case["planning"],
                    case["safe_buffer_remaining"]), start=1):
            writer.writerow([index, time_value, latency, remaining])

    counterfactual = straight_arm_gate_counterfactual(case)
    with (SOURCE_DATA / "straight_arm_gate_counterfactual.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["passage", "gate_y_m", "measured_uav_z_m",
                         "measured_theta_deg", "measured_phi_deg",
                         "measured_arm_clearance_m", "straight_arm_clearance_m",
                         "planning_arm_margin_m"])
        for gate_number, index in enumerate(counterfactual["indices"], start=1):
            writer.writerow([
                gate_number, counterfactual["gate_y"][gate_number - 1],
                case["positions"][index, 2], case["arm_deg"][index, 0],
                case["arm_deg"][index, 1],
                counterfactual["actual_clearance"][gate_number - 1],
                counterfactual["clearance"][gate_number - 1],
                case["limits"].arm_margin,
            ])

    with (SOURCE_DATA / "passage_vertical_feasibility.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["passage", "gate_y_m", "lower_top_m", "upper_bottom_m",
                         "straight_arm_uav_z_min_m", "body_safe_uav_z_max_m",
                         "robust_vertical_slack_m"])
        for passage, bounds in enumerate(passage_vertical_feasibility(case), start=1):
            writer.writerow([
                passage, bounds["y"], bounds["lower_top"], bounds["upper_bottom"],
                bounds["straight_uav_z_min"], bounds["body_uav_z_max"],
                bounds["robust_slack"],
            ])


def tex_escape(value):
    return str(value).replace("_", "\\_").replace("%", "\\%")


def write_tables(summary, rows, paired_stats):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Closed-loop comparison over 90 paired scenarios (450 trials). Continuous metrics are reported on each method's own successful trials.}",
        r"\label{tab:main_results}",
        r"\setlength{\tabcolsep}{3.1pt}",
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"Method & Success (95\% CI) & Task med. & Plan P95 & Deadline & Body/Arm P05 & Track RMSE & Jerk/m & Tip \\",
        r" &  & (s) & (s) & miss (\%) & (m) & (m) &  & (m) \\",
        r"\midrule",
    ]
    for method in METHODS:
        item = summary[method]
        label = LABELS[method]
        if method == "rolling_wb_v2":
            label = r"\textbf{Rolling-WB-V2.1}"
        values = (
            100.0 * item["success_rate"],
            100.0 * item["success_ci95_low"],
            100.0 * item["success_ci95_high"],
            item["task_time_median_s"], item["planning_p95_s"],
            100.0 * item["deadline_miss_rate"], item["body_clearance_p05_m"],
            item["arm_clearance_p05_m"], item["tracking_rmse_median_m"],
            item["jerk_energy_median"], item["tip_error_median_m"],
        )
        lines.append(
            "%s & %.1f [%.1f, %.1f] & %.2f & %.2f & %.2f & %.4f/%.4f & %.4f & %.4f & %.4f \\\\" %
            ((label,) + values)
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (TABLES / "table_main_results.tex").write_text("\n".join(lines), encoding="utf-8")

    difficulty = grouped_success(rows, "scenario_difficulty", DIFFICULTIES)
    family = grouped_success(rows, "scenario_family", FAMILIES)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Task success by difficulty and scenario family.}",
        r"\label{tab:success_breakdown}",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Group & Fixed & AET & WB-RRT* & Global & Ours \\", r"\midrule",
    ]
    for group in DIFFICULTIES:
        values = [100.0 * difficulty[group][method] for method in METHODS]
        lines.append("%s & %s \\\\" % (
            group.title(), " & ".join("%.1f" % value for value in values)))
    lines.append(r"\midrule")
    for group in FAMILIES:
        values = [100.0 * family[group][method] for method in METHODS]
        lines.append("%s & %s \\\\" % (
            FAMILY_LABELS[group], " & ".join("%.1f" % value for value in values)))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (TABLES / "table_success_breakdown.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Exact paired McNemar tests for task success.}",
        r"\label{tab:mcnemar}",
        r"\begin{tabular}{lrrc}", r"\toprule",
        r"Baseline & Ours only & Baseline only & $p$ \\", r"\midrule",
    ]
    for method in METHODS[:-1]:
        item = paired_stats["success_mcnemar"][method]
        lines.append("%s & %d & %d & %.2e \\\\" % (
            LABELS[method], item["ours_only_success"], item["baseline_only_success"],
            item["exact_two_sided_p"]))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (TABLES / "table_mcnemar.tex").write_text("\n".join(lines), encoding="utf-8")


def write_readme(case):
    text = """# SCI paper figure package

All figures are generated from the frozen `4f80214` experiment artifacts. PNG
files are 600 dpi; PDF files retain vector text and line work. Statistical
plots use all 90 paired scenarios. The representative case is the measured
PX4/Gazebo execution `multi_passage_hard_s009` (8 validated plans, 7 rolling
switches), not an offline interpolation.

## Figures

1. `fig01_success_and_failure_overview`: overall Wilson CI, difficulty/family
   success and retained failure outcomes.
2. `fig02_execution_quality_distributions`: planning, task time, clearance,
   tracking and jerk distributions on the 46 scenarios completed by all five
   methods (box: IQR; whiskers: 5th--95th percentile).  The paired common-success
   subset avoids comparing different methods on differently selected easy cases.
3. `fig03_paired_outcome_matrix`: all 90 paired binary outcomes.
4. `fig04_measured_whole_body_trajectory`: measured UAV/tool paths and PCC arm
   snapshots. Grey points are the audited RGB-D occupied voxels; grey solids
   are hidden Oracle geometry displayed only for post-hoc evaluation. In the
   side view, each passage overlays the measured bent arm (green) and a
   straight-arm counterfactual (red dashed) at exactly the same measured UAV
   pose. Red crosses and labels report the resulting physical collision;
   dashed lifted body envelopes show that raising the straight-arm vehicle
   leaves no robust vertical-feasibility interval. Four faint measured
   snapshots show pre-gate, inter-gate, and post-gate morphology transitions.
   Translucent blue and green regions show the UAV and arm safety envelopes,
   including map inflation and their respective planning margins.
5. `fig05_rolling_horizon_updates`: all eight accepted rolling trajectories
   overlaid with the measured execution.
6. `fig06_representative_execution_time_history`: measured progress, height,
   morphology, Oracle clearance, tracking/COM and replanning latency. Phi is
   masked when theta < 5 deg because the bending-plane angle is physically
   undefined for a straight continuum section.

## Tables and source data

- `tables/*.tex`: IEEE/LaTeX-ready tables using booktabs.
- `source_data/main_results.csv`: complete values behind the main table.
- `source_data/mcnemar_tests.csv`: exact paired success tests.
- `source_data/common_success_execution_metrics.csv`: the paired 46-scenario
  subset behind Fig. 2.
- `source_data/success_breakdown.csv`: values behind Fig. 1.
- `source_data/representative_measured_timeline.csv`: values behind Figs. 4/6.
- `source_data/rolling_window_latency.csv`: accepted-window latencies.
- `source_data/straight_arm_gate_counterfactual.csv`: measured-versus-straight
  arm clearance at the same UAV pose for each passage.
- `source_data/passage_vertical_feasibility.csv`: robust lower/upper UAV-center
  bounds showing whether raising a straight-arm UAV can satisfy both envelopes.

The plotted post-hoc minimum clearances are recomputed from the recorded
measured state and exact Oracle boxes using the same PCC/body geometry. They
are evaluation quantities and were not available to the planner.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main():
    style()
    rows = load_trials()
    summary = load_json(FINAL / "summary.json")
    paired_stats = load_json(FINAL / "paired_statistics.json")
    if len(rows) != 450:
        raise RuntimeError("expected 450 paired trial rows, got %d" % len(rows))
    case = representative_data()

    make_overview(rows, summary, paired_stats)
    make_continuous_metrics(rows)
    make_paired_matrix(rows)
    make_whole_body_trajectory(case)
    make_rolling_horizons(case)
    make_case_time_history(case)
    write_source_data(rows, case, summary, paired_stats)
    write_tables(summary, rows, paired_stats)
    write_readme(case)

    verification = {
        "paired_trial_rows": len(rows),
        "complete_pairs": len(rows) // len(METHODS),
        "common_success_pairs_for_figure_2": len(common_success_scenarios(rows)),
        "representative_run": str(case["run"]),
        "representative_samples": len(case["time"]),
        "representative_planning_windows": len(case["planning"]),
        "planning_deadline_s": 15.0,
        "certified_buffer_remaining_after_validation_min_s": float(
            np.nanmin(case["safe_buffer_remaining"])),
        "straight_arm_gate_clearance_m": [float(value) for value in
                                           straight_arm_gate_counterfactual(case)["clearance"]],
        "straight_arm_robust_vertical_slack_m": [
            float(item["robust_slack"]) for item in passage_vertical_feasibility(case)
        ],
        "recomputed_body_clearance_min_m": float(np.min(case["body_clearance"])),
        "recomputed_arm_clearance_min_m": float(np.min(case["arm_clearance"])),
        "audit_body_clearance_min_m": case["acceptance"]["measured_clearance"]["body_min"],
        "audit_arm_clearance_min_m": case["acceptance"]["measured_clearance"]["arm_min"],
        "generated_figures": sorted(path.name for path in FIGURES.glob("*")),
        "generated_tables": sorted(path.name for path in TABLES.glob("*")),
    }
    (OUT / "generation_report.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
