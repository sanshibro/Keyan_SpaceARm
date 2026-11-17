#!/usr/bin/env python3
"""Build handoff-continuity and dynamics-proxy evidence from frozen logs."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


OURS = "rolling_wb_v2"
CANDIDATE_FIELDS = (
    "position_gap_m", "yaw_gap_rad", "bending_gap_rad",
    "linear_velocity_gap_m_s", "bending_rate_gap_rad_s",
    "linear_acceleration_gap_m_s2",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def percentile(values, q):
    values = np.asarray(values, dtype=float)
    return None if not len(values) else float(np.percentile(values, q))


def summarize(values):
    return {
        "count": len(values), "median": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": None if not values else float(np.max(values)),
    }


def wrap_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def quaternion_yaw(quaternion):
    x, y, z, w = quaternion
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


def bending_vector_deg(arm_deg):
    arm_deg = np.asarray(arm_deg, dtype=float)
    theta = arm_deg[:, 0]
    phi = np.deg2rad(arm_deg[:, 1])
    return np.column_stack((theta * np.cos(phi), theta * np.sin(phi)))


def unique_switches(events):
    unique = {}
    for event in events:
        if (event.get("state") != "SWITCHED_ROLLING"
                or event.get("switch_continuity_diagnostics_version") != 2
                or event.get("trajectory_id") is None):
            continue
        identifier = int(event["trajectory_id"])
        if identifier not in unique or len(event) > len(unique[identifier]):
            unique[identifier] = event
    return [unique[key] for key in sorted(unique)]


def deduplicate(times, *arrays):
    times = np.asarray(times, dtype=float)
    order = np.argsort(times, kind="stable")
    times = times[order]
    arrays = [np.asarray(array)[order] for array in arrays]
    keep = np.r_[True, np.diff(times) > 1e-6]
    return (times[keep],) + tuple(array[keep] for array in arrays)


def command_series(run):
    commands = read_jsonl(run / "commands.jsonl")
    kinematics = read_jsonl(run / "command_kinematics.jsonl")
    commands = [row for row in commands
                if isinstance(row.get("arm_deg"), list)
                and len(row["arm_deg"]) == 2
                and all(value is not None for value in row["arm_deg"])
                and isinstance(row.get("position"), list)
                and isinstance(row.get("orientation"), list)]
    command_t = [row["stamp"] for row in commands]
    positions = [row["position"] for row in commands]
    yaw = [quaternion_yaw(row["orientation"]) for row in commands]
    bending = bending_vector_deg([row["arm_deg"] for row in commands])
    command_t, positions, yaw, bending = deduplicate(
        command_t, positions, np.unwrap(yaw), bending
    )
    kin_t = [row["stamp"] for row in kinematics]
    velocity = [row["linear_velocity"] for row in kinematics]
    yaw_rate = [row.get("yaw_rate", 0.0) for row in kinematics]
    kin_t, velocity, yaw_rate = deduplicate(kin_t, velocity, yaw_rate)
    lower = max(command_t[0], kin_t[0])
    upper = min(command_t[-1], kin_t[-1])
    dt = 0.05
    time = np.arange(lower, upper + 0.5 * dt, dt)
    velocity_u = np.column_stack([
        np.interp(time, kin_t, velocity[:, axis]) for axis in range(3)
    ])
    acceleration = savgol_filter(
        velocity_u, 11, 3, deriv=1, delta=dt, axis=0, mode="interp"
    )
    jerk = savgol_filter(
        velocity_u, 11, 3, deriv=2, delta=dt, axis=0, mode="interp"
    )
    position_u = np.column_stack([
        np.interp(time, command_t, positions[:, axis]) for axis in range(3)
    ])
    yaw_u = np.interp(time, command_t, yaw)
    bending_u = np.column_stack([
        np.interp(time, command_t, bending[:, axis]) for axis in range(2)
    ])
    bending_rate = savgol_filter(
        bending_u, 11, 3, deriv=1, delta=dt, axis=0, mode="interp"
    )
    return {
        "time": time, "position": position_u, "yaw": yaw_u,
        "bending_deg": bending_u, "velocity": velocity_u,
        "acceleration": acceleration, "jerk": jerk,
        "bending_rate_deg_s": bending_rate, "dt": dt,
    }


def adjacent_gap(series, stamp):
    time = series["time"]
    right = int(np.searchsorted(time, stamp, side="left"))
    right = min(max(right, 1), len(time) - 1)
    left = right - 1
    return {
        "issued_position_step_m": float(np.linalg.norm(
            series["position"][right] - series["position"][left])),
        "issued_yaw_step_rad": abs(wrap_angle(
            series["yaw"][right] - series["yaw"][left])),
        "issued_bending_step_deg": float(np.linalg.norm(
            series["bending_deg"][right] - series["bending_deg"][left])),
        "issued_velocity_step_m_s": float(np.linalg.norm(
            series["velocity"][right] - series["velocity"][left])),
        "issued_acceleration_step_m_s2": float(np.linalg.norm(
            series["acceleration"][right] - series["acceleration"][left])),
        "issued_bending_rate_step_deg_s": float(np.linalg.norm(
            series["bending_rate_deg_s"][right]
            - series["bending_rate_deg_s"][left])),
    }


def handoff_evidence(result_root):
    records = []
    run_cache = {}
    for path in result_root.rglob("acceptance_report.json"):
        try:
            acceptance = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if acceptance.get("method") != OURS:
            continue
        run = path.parent
        events = unique_switches(read_jsonl(run / "events.jsonl"))
        if not events:
            continue
        series = command_series(run)
        run_cache[str(run.resolve())] = (series, events)
        for event in events:
            bridge = event.get("switch_bridge", {})
            mode = ("C2" if bridge.get("status") == "SAFE" else "C1")
            record = {
                "run_dir": str(run.resolve()),
                "trajectory_id": int(event["trajectory_id"]),
                "switch_stamp": float(event.get("actual_start", event["stamp"])),
                "handoff_mode": mode,
                "bridge_duration_s": float(bridge.get("duration_s", 0.0) or 0.0),
            }
            for field in CANDIDATE_FIELDS:
                record["candidate_" + field] = float(event[field])
            record.update(adjacent_gap(series, record["switch_stamp"]))
            records.append(record)
    if len(records) != 391:
        raise RuntimeError("expected 391 diagnosed handoffs, got %d" % len(records))
    fields = [key for key in records[0] if key.startswith(("candidate_", "issued_"))]
    statistics = {field: summarize([row[field] for row in records]) for field in fields}
    return records, statistics, run_cache


def select_representative_switch(output, records, run_cache):
    generation = read_json(output / "generation_report.json")
    representative = str(Path(generation["representative_run"]).resolve())
    candidates = [row for row in records
                  if row["run_dir"] == representative and row["handoff_mode"] == "C2"]
    if not candidates:
        candidates = [row for row in records if row["run_dir"] == representative]
    if not candidates:
        raise RuntimeError("representative run has no diagnosed switch")
    selected = sorted(candidates, key=lambda row: row["trajectory_id"])[0]
    return selected, run_cache[representative][0]


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.2,
        "axes.titlesize": 8.2, "axes.labelsize": 7.4,
        "legend.fontsize": 6.4, "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7, "axes.grid": True,
        "grid.alpha": 0.25, "savefig.dpi": 600,
    })


def save(fig, figures, stem):
    fig.savefig(str(figures / (stem + ".png")), dpi=600, bbox_inches="tight")
    fig.savefig(str(figures / (stem + ".pdf")), bbox_inches="tight")
    plt.close(fig)


def make_handoff_figure(output, selected, series):
    figures = output / "figures"
    stamp = selected["switch_stamp"]
    mask = (series["time"] >= stamp - 2.5) & (series["time"] <= stamp + 3.0)
    t = series["time"][mask] - stamp
    position = series["position"][mask]
    position = position - position[np.argmin(np.abs(t))]
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.15), sharex=True)
    panels = (
        (position, ("x", "y", "z"), "Relative position (m)", "Issued position"),
        (series["velocity"][mask], ("vx", "vy", "vz"), "Velocity (m/s)", "Issued velocity"),
        (series["acceleration"][mask], ("ax", "ay", "az"), "Acceleration (m/s²)", "Filtered acceleration"),
        (np.linalg.norm(series["jerk"][mask], axis=1)[:, None], ("norm",),
         "Jerk magnitude (m/s³)", "Executed-command jerk"),
        (series["bending_deg"][mask], ("bx", "by"), "Bending vector (deg)",
         "Issued morphology vector"),
        (series["bending_rate_deg_s"][mask], ("dbx", "dby"),
         "Bending rate (deg/s)", "Issued morphology rate"),
    )
    colors = ("#0072B2", "#D55E00", "#009E73")
    for index, (ax, (values, labels, ylabel, title)) in enumerate(zip(axes.ravel(), panels)):
        for column, label in enumerate(labels):
            ax.plot(t, values[:, column], lw=1.0, color=colors[column], label=label)
        ax.axvline(0.0, color="#222222", linestyle="--", lw=0.8)
        duration = selected["bridge_duration_s"]
        if duration > 0.0:
            ax.axvspan(0.0, duration, color="#E69F00", alpha=0.10)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, ncol=min(3, len(labels)), loc="best")
        ax.text(0.015, 0.96, "(%s)" % chr(ord("a") + index),
                transform=ax.transAxes, va="top", fontweight="bold")
        if index >= 3:
            ax.set_xlabel("Time from handoff (s)")
    fig.suptitle(
        "Closed-loop simulated command continuity around a validated handoff",
        fontsize=9.0, fontweight="bold"
    )
    fig.text(0.5, 0.018,
             "Dashed line: handoff request; shaded interval: finite-duration C2 bridge",
             ha="center", va="bottom", fontsize=6.4)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.89, bottom=0.15,
                        wspace=0.34, hspace=0.40)
    save(fig, figures, "fig07_handoff_continuity")


def dynamics_series(run):
    events = read_jsonl(run / "events.jsonl")
    commands = read_jsonl(run / "command_kinematics.jsonl")
    dynamics = read_jsonl(run / "flight_dynamics.jsonl")
    starts = [row["actual_start"] for row in events
              if row.get("state") == "START_SCHEDULED"]
    stops = [row.get("logger_stamp", row.get("stamp")) for row in events
             if row.get("state") == "TASK_COMPLETE"]
    lower, upper = min(starts), max(stops)
    commands = [row for row in commands if lower <= row["stamp"] <= upper]
    imu = [row for row in dynamics
           if row.get("kind") == "imu" and lower <= row["stamp"] <= upper]
    targets = [row for row in dynamics
               if row.get("kind") == "attitude_target"
               and lower <= row["stamp"] <= upper]
    source_t = [row["stamp"] for row in commands]
    source_v = [row["linear_velocity"] for row in commands]
    source_t, source_v = deduplicate(source_t, source_v)
    dt = 0.05
    time = np.arange(source_t[0], source_t[-1] + 0.5 * dt, dt)
    velocity = np.column_stack([
        np.interp(time, source_t, source_v[:, axis]) for axis in range(3)
    ])
    acceleration = savgol_filter(
        velocity, 11, 3, deriv=1, delta=dt, axis=0, mode="interp"
    )
    gravity = 9.80665
    force = acceleration + np.asarray([0.0, 0.0, gravity])
    planned_thrust = np.linalg.norm(force, axis=1) / gravity
    planned_tilt = np.arctan2(np.linalg.norm(force[:, :2], axis=1),
                              np.maximum(force[:, 2], 1e-6))
    imu_t = np.asarray([row["stamp"] for row in imu], dtype=float)
    actual_tilt_raw = np.hypot([row["roll_rad"] for row in imu],
                               [row["pitch_rad"] for row in imu])
    actual_force_raw = np.asarray([
        np.linalg.norm(row["linear_acceleration_m_s2"]) / gravity for row in imu
    ])
    target_t = np.asarray([row["stamp"] for row in targets], dtype=float)
    target_tilt_raw = np.hypot([row["roll_rad"] for row in targets],
                               [row["pitch_rad"] for row in targets])
    actual_tilt = np.interp(time, imu_t, actual_tilt_raw)
    actual_force = np.interp(time, imu_t, actual_force_raw)
    force_window = min(41, len(actual_force) if len(actual_force) % 2 else len(actual_force) - 1)
    actual_force_lowpass = savgol_filter(
        actual_force, max(5, force_window), 2, mode="interp"
    )
    target_tilt = np.interp(time, target_t, target_tilt_raw)
    rho = float(spearmanr(target_tilt, actual_tilt)[0])
    thrust_error = planned_thrust - actual_force_lowpass
    thrust_rho = float(spearmanr(planned_thrust, actual_force_lowpass)[0])
    return {
        "run": run, "time": time - time[0], "planned_thrust": planned_thrust,
        "actual_force": actual_force, "actual_force_lowpass": actual_force_lowpass,
        "planned_tilt": planned_tilt,
        "target_tilt": target_tilt, "actual_tilt": actual_tilt,
        "summary": {
            "planned_thrust_max": float(np.max(planned_thrust)),
            "imu_specific_force_max": float(np.max(actual_force)),
            "imu_specific_force_lowpass_max": float(np.max(actual_force_lowpass)),
            "thrust_proxy_rmse": float(np.sqrt(np.mean(thrust_error ** 2))),
            "thrust_proxy_mae": float(np.mean(np.abs(thrust_error))),
            "thrust_proxy_spearman_rho": thrust_rho,
            "planned_tilt_max_deg": float(np.rad2deg(np.max(planned_tilt))),
            "px4_target_tilt_max_deg": float(np.rad2deg(np.max(target_tilt))),
            "imu_tilt_max_deg": float(np.rad2deg(np.max(actual_tilt))),
            "target_actual_tilt_rmse_deg": float(np.rad2deg(np.sqrt(
                np.mean((target_tilt - actual_tilt) ** 2)))),
            "target_actual_spearman_rho": rho,
        },
    }


def dynamics_evidence(dynamics_root):
    runs = sorted({path.parent for path in dynamics_root.rglob("flight_dynamics.jsonl")
                   if (path.parent / "command_kinematics.jsonl").is_file()})
    if len(runs) != 3:
        raise RuntimeError("expected three frozen dynamics runs, got %d" % len(runs))
    return [dynamics_series(run) for run in runs]


def scenario_label(run):
    name = run.name.split("__rolling_wb_v2__", 1)[0]
    if name.startswith("passage_sweep"):
        return "Passage-Sweep"
    if name.startswith("multi_passage"):
        return "Multi-Passage"
    return "Clutter-Reach"


def make_dynamics_figure(output, cases):
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.0), sharex="col")
    for column, case in enumerate(cases):
        t = case["time"]
        ax = axes[0, column]
        ax.plot(t, case["planned_thrust"], color="#0072B2", lw=0.9,
                label=r"Planning proxy $r_T$")
        ax.plot(t, case["actual_force_lowpass"], color="#D55E00", lw=0.85,
                label="Low-pass simulated IMU / g")
        ax.axhline(1.0, color="#777777", linestyle=":", lw=0.7)
        ax.set_ylabel("Thrust/load ratio")
        ax.set_title(scenario_label(case["run"]))
        ax.text(0.98, 0.94, "$r_T^{max}$=%.4f\nRMSE=%.4f, $\\rho$=%.2f" %
                (case["summary"]["planned_thrust_max"],
                 case["summary"]["thrust_proxy_rmse"],
                 case["summary"]["thrust_proxy_spearman_rho"]), transform=ax.transAxes,
                ha="right", va="top", fontsize=6.1)
        if column == 0:
            ax.legend(frameon=False, loc="lower left")
        ax = axes[1, column]
        ax.plot(t, np.rad2deg(case["planned_tilt"]), color="#0072B2", lw=0.9,
                label=r"Planning proxy $\gamma$")
        ax.plot(t, np.rad2deg(case["target_tilt"]), color="#009E73", lw=0.8,
                label="PX4 target")
        ax.plot(t, np.rad2deg(case["actual_tilt"]), color="#D55E00", lw=0.7,
                alpha=0.82, label="Simulated IMU actual")
        ax.set_xlabel("Elapsed execution time (s)")
        ax.set_ylabel("Tilt (deg)")
        ax.text(0.98, 0.94, "$\\gamma_{max}$=%.3f$^\\circ$\nlimit 15$^\\circ$" %
                case["summary"]["planned_tilt_max_deg"], transform=ax.transAxes,
                ha="right", va="top", fontsize=6.1)
        if column == 0:
            ax.legend(frameon=False, loc="lower left")
    for index, ax in enumerate(axes.ravel()):
        ax.text(0.015, 0.96, "(%s)" % chr(ord("a") + index),
                transform=ax.transAxes, va="top", fontweight="bold")
    fig.suptitle("Dynamics-aware proxies and closed-loop simulated PX4 response",
                 fontsize=9.0, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.995, top=0.88, bottom=0.11,
                        wspace=0.34, hspace=0.30)
    save(fig, output / "figures", "fig08_dynamics_proxy_validation")


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tables(output, statistics, cases, result_root):
    rows = (
        ("Position", "m", "candidate_position_gap_m", "issued_position_step_m"),
        ("Yaw", "deg", "candidate_yaw_gap_rad", "issued_yaw_step_rad"),
        ("Bending vector", "deg", "candidate_bending_gap_rad", "issued_bending_step_deg"),
        ("Linear velocity", "m/s", "candidate_linear_velocity_gap_m_s", "issued_velocity_step_m_s"),
        ("Linear acceleration", "m/s$^2$", "candidate_linear_acceleration_gap_m_s2", "issued_acceleration_step_m_s2"),
    )
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Command-continuity evidence over all 391 rolling handoffs. Candidate gaps are evaluated before an optional C2 bridge; issued-command steps are adjacent 50-ms closed-loop simulated commands and are not claimed as an analytic proof of strict $C^2$ continuity.}",
        r"\label{tab:handoff_continuity}",
        r"\begin{tabular}{llrrr|rrr}", r"\toprule",
        r"Metric & Unit & \multicolumn{3}{c|}{Pre-bridge candidate gap} & \multicolumn{3}{c}{Issued-command step} " + r"\\",
        r" & & Median & P95 & Max & Median & P95 & Max " + r"\\", r"\midrule",
    ]
    for label, unit, candidate, issued in rows:
        left, right = statistics[candidate], statistics[issued]
        factor_left = 180.0 / math.pi if candidate.endswith("_rad") else 1.0
        factor_right = 180.0 / math.pi if issued.endswith("_rad") else 1.0
        lines.append("%s & %s & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f \\\\" % (
            label, unit, left["median"] * factor_left, left["p95"] * factor_left,
            left["max"] * factor_left, right["median"] * factor_right,
            right["p95"] * factor_right, right["max"] * factor_right))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (output / "tables/table_handoff_continuity.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Dynamics-proxy correspondence in three closed-loop simulated PX4 tasks. Thrust errors compare $r_T$ with low-pass IMU/g in the quasi-hover regime; IMU specific force is a collective-load proxy, not direct motor thrust.}",
        r"\label{tab:dynamics_proxy}", r"\scriptsize",
        r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
        r"Scenario & $r_T^{max}$ & LP IMU/g max & RMSE & MAE & $\rho_T$ & $\gamma^{max}$ & Target max & IMU max & Tilt RMSE & $\rho_\gamma$ " + r"\\",
        r" & & & & & & (deg) & (deg) & (deg) & (deg) & " + r"\\", r"\midrule",
    ]
    for case in cases:
        item = case["summary"]
        lines.append("%s & %.4f & %.4f & %.4f & %.4f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\" % (
            scenario_label(case["run"]), item["planned_thrust_max"],
            item["imu_specific_force_lowpass_max"], item["thrust_proxy_rmse"],
            item["thrust_proxy_mae"], item["thrust_proxy_spearman_rho"],
            item["planned_tilt_max_deg"],
            item["px4_target_tilt_max_deg"], item["imu_tilt_max_deg"],
            item["target_actual_tilt_rmse_deg"], item["target_actual_spearman_rho"]))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (output / "tables/table_dynamics_proxy.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    paired = read_json(result_root / "final_evidence/common_success_statistics.json")
    metric_labels = {
        "task_time_s": "Task time (s)", "planning_time_s": "Planning time (s)",
        "body_clearance_m": "Body clearance (m)",
        "arm_clearance_m": "Arm clearance (m)",
        "tracking_rmse_m": "Tracking RMSE (m)",
        "com_offset_max_m": "COM offset (m)", "jerk_energy": "Jerk/m",
        "tip_error_m": "Tip error (m)",
    }
    baseline_labels = {
        "fixed_morphology": "Fixed", "aet_sequential_minjerk": "AET-Seq.",
        "informed_rrtstar_wb": "WB-RRT*", "global_batch_wb": "Global-WB",
    }
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\scriptsize",
        r"\caption{Paired continuous-metric comparisons on the 54 scenarios completed by all five methods. Delta is Rolling-WB-V2.1 minus baseline; intervals are deterministic paired-bootstrap 95\% CIs. Bootstrap intervals estimate the paired median difference, whereas Holm-adjusted Wilcoxon tests concern signed ranks; their conclusions therefore need not coincide.}",
        r"\label{tab:paired_continuous}",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Metric & Baseline & Median delta & Bootstrap 95\% CI & Holm $p$ & Rank-biserial $r$ " + r"\\",
        r"\midrule",
    ]
    previous = None
    for item in paired["wilcoxon_holm"]:
        metric = metric_labels[item["metric"]]
        if previous is not None and item["metric"] != previous:
            lines.append(r"\addlinespace")
        lines.append("%s & %s & %.5g & [%.5g, %.5g] & %.3g & %.3f \\\\" % (
            metric, baseline_labels[item["baseline"]],
            item["median_delta_ours_minus_baseline"],
            item["median_delta_bootstrap95_low"],
            item["median_delta_bootstrap95_high"], item["holm_p"],
            item["matched_rank_biserial_ours_minus_baseline"]))
        previous = item["metric"]
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (output / "tables/table_paired_continuous.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dynamics-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    for directory in (output / "figures", output / "tables", output / "source_data"):
        directory.mkdir(parents=True, exist_ok=True)
    style()
    records, statistics, run_cache = handoff_evidence(args.result_root.resolve())
    selected, series = select_representative_switch(output, records, run_cache)
    make_handoff_figure(output, selected, series)
    cases = dynamics_evidence(args.dynamics_root.resolve())
    make_dynamics_figure(output, cases)
    write_csv(output / "source_data/handoff_continuity_samples.csv", records)
    dynamics_rows = []
    for case in cases:
        dynamics_rows.append({"scenario": scenario_label(case["run"]),
                              "run_dir": str(case["run"].resolve()),
                              **case["summary"]})
    write_csv(output / "source_data/dynamics_proxy_summary.csv", dynamics_rows)
    write_tables(output, statistics, cases, args.result_root.resolve())
    report = {
        "handoff_count": len(records),
        "c1_count": sum(row["handoff_mode"] == "C1" for row in records),
        "c2_count": sum(row["handoff_mode"] == "C2" for row in records),
        "candidate_and_issued_statistics": statistics,
        "representative_handoff": selected,
        "dynamics_runs": dynamics_rows,
        "claim_boundary": (
            "Dynamics quantities are conservative planning/collective-load proxies; "
            "no direct motor-thrust or complete dynamics-feasibility claim is made."
        ),
    }
    (output / "supplementary_evidence.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    position = statistics["issued_position_step_m"]
    velocity = statistics["issued_velocity_step_m_s"]
    acceleration = statistics["issued_acceleration_step_m_s2"]
    bending = statistics["issued_bending_step_deg"]
    markdown = """# Rolling-WB-V2.1 E3 后处理证据补充报告

## 1. 失败统计

V2.1 的唯一失败 `repeated_planning_rejection` 已归入 `No path / gate`
类别并在图 1(d) 显式标注为 1。因而总成功数 89、难度分组
`30+30+29`、场景族分组 `30+30+29` 与 `90-1` 完全一致。

## 2. 接管连续性

现有闭环日志共重建 {handoffs} 次接管，其中 C1 {c1} 次、C2 {c2}
次。相邻 50 ms 实际下发仿真指令的 median/P95/max 为：

- 位置步长：{pmed:.4f}/{p95:.4f}/{pmax:.4f} m；
- 速度步长：{vmed:.4f}/{v95:.4f}/{vmax:.4f} m/s；
- 滤波加速度步长：{amed:.4f}/{a95:.4f}/{amax:.4f} m/s²；
- 弯曲向量步长：{bmed:.3f}/{b95:.3f}/{bmax:.3f} deg。

图 7 显示一次采用 C2 桥接的验证接管前后的位置、速度、加速度、jerk、
弯曲向量和弯曲速率。黑色虚线表示接管请求，黄色区域表示有限时长桥接，
位置曲线相对接管参考绘制，因此桥接期间的变化表示轨迹继续推进而非瞬时
位置跳变。这些数值是有限采样周期下的相邻指令步长，不宣称为解析意义的
严格 C2 证明。

## 3. 动力学代理

图 8 和表 `table_dynamics_proxy.tex` 同时报告三类任务的规划
推力比 $r_T$、规划倾角 $\\gamma$、PX4 目标倾角、仿真 IMU 实际倾角和
低通 IMU 比力。推力代理误差和相关系数见 `table_dynamics_proxy.tex`；
该证据只覆盖准悬停范围。姿态目标与实际倾角的 Spearman 相关系数为
0.909--0.967，倾角 RMSE 为 0.061--0.107 deg。IMU 比力仅作集体负载
代理，不当作实测电机推力，也不外推到完整飞行包线。

## 4. 统计与表述

主表明确为各方法自身成功集，图 2 为 54 个五方法共同成功配对场景。
连续指标补充 Friedman、Wilcoxon--Holm、配对中位差和 10000 次配对
bootstrap 95% CI。所有仿真图均使用 `Closed-loop simulated`
措辞；规划图同时显示 15 s deadline 与新候选验证完成后的旧安全缓存
剩余时间；形态图使用 $b_x=\\theta\\cos\\phi$ 和
$b_y=\\theta\\sin\\phi$ 避免 $\\phi$ 的 $\\pm180^\\circ$ 包络跳变。
Bootstrap 中位差区间与 Holm 校正 Wilcoxon 检验针对的统计量不同，两者
结论不要求完全一致。COM 只表述为与基线可比；较小但满足阈值的机体净空
表述为空间利用，不表述为最大安全裕度。
""".format(
        handoffs=len(records), c1=report["c1_count"], c2=report["c2_count"],
        pmed=position["median"], p95=position["p95"], pmax=position["max"],
        vmed=velocity["median"], v95=velocity["p95"], vmax=velocity["max"],
        amed=acceleration["median"], a95=acceleration["p95"],
        amax=acceleration["max"], bmed=bending["median"],
        b95=bending["p95"], bmax=bending["max"],
    )
    (output / "POSTPROCESS_EVIDENCE_REPORT_CN.md").write_text(
        markdown, encoding="utf-8"
    )
    print(json.dumps({"result": "PASS", "handoffs": len(records),
                      "dynamics_runs": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
