#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def resolve_csv(path):
    if os.path.isdir(path):
        candidate = os.path.join(path, "orb_strain_sync.csv")
        if os.path.exists(candidate):
            return candidate
    return path


def causal_moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    out = np.empty_like(values, dtype=float)
    csum = np.cumsum(np.insert(values, 0, 0.0))
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        out[idx] = (csum[idx + 1] - csum[start]) / (idx - start + 1)
    return out


def read_data(csv_path, channel, max_theta_age):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row.get("strain_channel", channel)) != channel:
                    continue
                if int(float(row.get("usable", "1") or 1)) != 1:
                    continue
                theta_age_text = row.get("theta_age", "")
                if theta_age_text and float(theta_age_text) > max_theta_age:
                    continue
                rel_time = float(row["rel_time"])
                strain = float(row["strain_value"])
                theta_text = row.get("orb_theta_deg", "")
                theta = None if theta_text == "" else float(theta_text)
            except (KeyError, TypeError, ValueError):
                continue
            rows.append((rel_time, strain, theta))
    return rows


def metrics(y_true, y_pred):
    err = y_pred - y_true
    return {
        "mae_deg": float(np.mean(np.abs(err))),
        "rmse_deg": float(np.sqrt(np.mean(err ** 2))),
        "max_abs_error_deg": float(np.max(np.abs(err))),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate a strain-to-theta model on a new run.")
    parser.add_argument("model_json")
    parser.add_argument("csv_or_run_dir")
    parser.add_argument("--channel", type=int, default=None)
    parser.add_argument("--max-theta-age", type=float, default=0.2)
    parser.add_argument(
        "--auto-baseline",
        action="store_true",
        help="use this run's initial zero-pose strain baseline instead of the training baseline",
    )
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=20.0,
        help="initial seconds used by --auto-baseline",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    csv_path = resolve_csv(args.csv_or_run_dir)
    if not os.path.exists(args.model_json):
        print(f"Model not found: {args.model_json}", file=sys.stderr)
        return 1
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    with open(args.model_json, encoding="utf-8") as f:
        model = json.load(f)
    channel = int(model.get("channel", 1) if args.channel is None else args.channel)
    rows = read_data(csv_path, channel, args.max_theta_age)
    if len(rows) < 5:
        print(f"Not enough usable rows: {len(rows)}", file=sys.stderr)
        return 2

    t = np.asarray([r[0] for r in rows], dtype=float)
    strain = np.asarray([r[1] for r in rows], dtype=float)
    theta_values = [r[2] for r in rows]
    has_theta = all(v is not None for v in theta_values)
    theta = np.asarray(theta_values, dtype=float) if has_theta else None

    smooth_window = int(model.get("smooth_window", 1))
    coeffs = np.asarray(model["coefficients_high_to_low"], dtype=float)
    model_strain = causal_moving_average(strain, smooth_window)
    if args.auto_baseline:
        mask = t <= (float(t[0]) + args.baseline_seconds)
        if int(np.sum(mask)) < 5:
            print("Not enough initial samples for --auto-baseline.", file=sys.stderr)
            return 2
        baseline = float(np.median(model_strain[mask]))
        baseline_source = f"auto_initial_{args.baseline_seconds:g}s"
    else:
        baseline = float(model["baseline"])
        baseline_source = "model"
    pred = np.polyval(coeffs, model_strain - baseline)

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(csv_path), "strain_theta_validation")
    os.makedirs(out_dir, exist_ok=True)

    pred_csv = os.path.join(out_dir, "validation_predictions.csv")
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rel_time", "strain_value", "model_strain", "theta_true_deg", "theta_pred_deg", "error_deg"])
        for idx in range(len(t)):
            true_text = "" if theta is None else f"{theta[idx]:.6f}"
            err_text = "" if theta is None else f"{pred[idx] - theta[idx]:.6f}"
            writer.writerow([f"{t[idx]:.6f}", f"{strain[idx]:.6f}", f"{model_strain[idx]:.6f}", true_text, f"{pred[idx]:.6f}", err_text])

    plot_path = os.path.join(out_dir, "validation.png")
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    if theta is not None:
        ax.plot(t, theta, "k.", markersize=2, label="ORB theta")
    ax.plot(t, pred, "tab:red", linewidth=1.3, label="predicted theta")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("theta (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_json": os.path.abspath(args.model_json),
        "source_csv": os.path.abspath(csv_path),
        "samples": int(len(t)),
        "smooth_window": smooth_window,
        "baseline": baseline,
        "baseline_source": baseline_source,
    }
    if theta is not None:
        report["metrics"] = metrics(theta, pred)
    report_path = os.path.join(out_dir, "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"samples={len(t)}")
    print(f"baseline={baseline:.6f} source={baseline_source}")
    if theta is not None:
        m = report["metrics"]
        print(f"mae={m['mae_deg']:.3f} deg rmse={m['rmse_deg']:.3f} deg max_abs={m['max_abs_error_deg']:.3f} deg")
    print(f"Saved predictions: {pred_csv}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
