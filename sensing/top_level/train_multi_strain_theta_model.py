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


def read_dataset(csv_path, channel, max_theta_age, require_usable):
    strain = []
    theta = []
    rel_time = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row.get("strain_channel", channel)) != channel:
                    continue
                if require_usable and int(float(row.get("usable", "0") or 0)) != 1:
                    continue
                theta_age_text = row.get("theta_age", "")
                if theta_age_text and float(theta_age_text) > max_theta_age:
                    continue
                strain.append(float(row["strain_value"]))
                theta.append(float(row["orb_theta_deg"]))
                rel_time.append(float(row["rel_time"]))
            except (KeyError, TypeError, ValueError):
                continue
    return {
        "csv_path": csv_path,
        "strain": np.asarray(strain, dtype=float),
        "theta": np.asarray(theta, dtype=float),
        "rel_time": np.asarray(rel_time, dtype=float),
    }


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


def estimate_baseline(rel_time, smoothed, baseline_seconds):
    if len(smoothed) == 0:
        raise ValueError("empty dataset")
    t0 = float(rel_time[0])
    mask = rel_time <= t0 + baseline_seconds
    if int(np.sum(mask)) < 5:
        count = max(5, min(200, len(smoothed)))
        return float(np.median(smoothed[:count]))
    return float(np.median(smoothed[mask]))


def build_features(datasets, smooth_window, baseline_seconds, test_fraction):
    rows = []
    baselines = []
    sample_offset = 0
    for run_id, ds in enumerate(datasets):
        smoothed = causal_moving_average(ds["strain"], smooth_window)
        baseline = estimate_baseline(ds["rel_time"], smoothed, baseline_seconds)
        baselines.append(baseline)
        n = len(smoothed)
        test_n = max(1, int(round(n * test_fraction)))
        split_at = max(1, n - test_n)
        for idx in range(n):
            rows.append(
                {
                    "run_id": run_id,
                    "csv_path": ds["csv_path"],
                    "rel_time": float(ds["rel_time"][idx]),
                    "sample_index": sample_offset + idx,
                    "strain": float(ds["strain"][idx]),
                    "smoothed": float(smoothed[idx]),
                    "baseline": baseline,
                    "delta": float(smoothed[idx] - baseline),
                    "theta": float(ds["theta"][idx]),
                    "split": "train" if idx < split_at else "test",
                }
            )
        sample_offset += n
    return rows, baselines


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {"mae_deg": mae, "rmse_deg": rmse, "r2": r2}


def fit_and_score(rows, degree):
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    x_train = np.asarray([row["delta"] for row in train_rows], dtype=float)
    y_train = np.asarray([row["theta"] for row in train_rows], dtype=float)
    x_test = np.asarray([row["delta"] for row in test_rows], dtype=float)
    y_test = np.asarray([row["theta"] for row in test_rows], dtype=float)
    coeffs = np.polyfit(x_train, y_train, deg=degree)
    return {
        "degree": degree,
        "coeffs": coeffs,
        "train_metrics": metrics(y_train, np.polyval(coeffs, x_train)),
        "test_metrics": metrics(y_test, np.polyval(coeffs, x_test)),
    }


def save_outputs(out_dir, args, datasets, rows, baselines, best, candidates):
    os.makedirs(out_dir, exist_ok=True)
    coeffs = best["coeffs"]
    model = {
        "type": "numpy_polyfit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_csvs": [os.path.abspath(ds["csv_path"]) for ds in datasets],
        "channel": args.channel,
        "degree": int(best["degree"]),
        "smooth_window": int(best["smooth_window"]),
        "baseline": float(np.median(baselines)),
        "per_run_baselines": [float(v) for v in baselines],
        "feature": "causal_moving_average(strain_value, smooth_window) - current_run_or_live_baseline",
        "coefficients_high_to_low": [float(c) for c in coeffs],
        "prediction": "theta_deg = polyval(coefficients_high_to_low, smoothed_strain - live_baseline)",
        "train_metrics": best["train_metrics"],
        "test_metrics": best["test_metrics"],
        "candidate_summary": candidates,
        "baseline_seconds": float(args.baseline_seconds),
    }
    model_path = os.path.join(out_dir, "strain_theta_model.json")
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)

    pred_csv = os.path.join(out_dir, "predictions.csv")
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_id",
                "rel_time",
                "sample_index",
                "split",
                "strain_value",
                "smoothed_strain",
                "run_baseline",
                "model_delta_strain",
                "theta_true_deg",
                "theta_pred_deg",
                "error_deg",
                "csv_path",
            ]
        )
        for row in rows:
            pred = float(np.polyval(coeffs, row["delta"]))
            writer.writerow(
                [
                    row["run_id"],
                    f"{row['rel_time']:.6f}",
                    row["sample_index"],
                    row["split"],
                    f"{row['strain']:.6f}",
                    f"{row['smoothed']:.6f}",
                    f"{row['baseline']:.6f}",
                    f"{row['delta']:.6f}",
                    f"{row['theta']:.6f}",
                    f"{pred:.6f}",
                    f"{pred - row['theta']:.6f}",
                    row["csv_path"],
                ]
            )

    plot_path = os.path.join(out_dir, "fit.png")
    x = np.asarray([row["delta"] for row in rows], dtype=float)
    y = np.asarray([row["theta"] for row in rows], dtype=float)
    idx = np.asarray([row["sample_index"] for row in rows], dtype=float)
    pred = np.polyval(coeffs, x)
    fig, (ax_time, ax_fit) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    ax_time.plot(idx, y, "k.", markersize=2, label="ORB theta")
    ax_time.plot(idx, pred, "tab:red", linewidth=1.0, label="predicted theta")
    ax_time.set_xlabel("sample index across runs")
    ax_time.set_ylabel("theta (deg)")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc="upper left")
    order = np.argsort(x)
    ax_fit.plot(x, y, "k.", markersize=2, alpha=0.45, label="samples")
    ax_fit.plot(x[order], pred[order], "tab:red", linewidth=1.5, label=f"degree {best['degree']} fit")
    ax_fit.set_xlabel("smoothed strain - per-run baseline")
    ax_fit.set_ylabel("theta (deg)")
    ax_fit.grid(True, alpha=0.3)
    ax_fit.legend(loc="upper left")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return model_path, pred_csv, plot_path


def main():
    parser = argparse.ArgumentParser(description="Train a strain-to-theta model from multiple synchronized runs.")
    parser.add_argument("csv_or_run_dirs", nargs="+", help="one or more orb_strain_sync.csv files or run directories")
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--smooth-window", type=int, default=None)
    parser.add_argument("--no-auto", action="store_true", help="disable candidate search; use --degree and --smooth-window")
    parser.add_argument("--baseline-seconds", type=float, default=5.0)
    parser.add_argument("--max-theta-age", type=float, default=0.2)
    parser.add_argument("--allow-unusable", action="store_true")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", default="strain_theta_model_multi")
    args = parser.parse_args()

    datasets = []
    for path in args.csv_or_run_dirs:
        csv_path = resolve_csv(path)
        if not os.path.exists(csv_path):
            print(f"CSV not found: {csv_path}", file=sys.stderr)
            return 1
        ds = read_dataset(csv_path, args.channel, args.max_theta_age, require_usable=not args.allow_unusable)
        if len(ds["strain"]) < 20:
            print(f"Skipping {csv_path}: not enough usable samples ({len(ds['strain'])})", file=sys.stderr)
            continue
        datasets.append(ds)
    if not datasets:
        print("No usable datasets.", file=sys.stderr)
        return 2

    if args.no_auto:
        degrees = [3 if args.degree is None else args.degree]
        windows = [3 if args.smooth_window is None else args.smooth_window]
    else:
        degrees = [1, 2, 3] if args.degree is None else [args.degree]
        windows = [1, 3, 5, 9] if args.smooth_window is None else [args.smooth_window]

    results = []
    by_key = {}
    for smooth_window in windows:
        rows, baselines = build_features(datasets, smooth_window, args.baseline_seconds, args.test_fraction)
        for degree in degrees:
            if len(rows) < max(20, degree + 3):
                continue
            result = fit_and_score(rows, degree)
            result["smooth_window"] = smooth_window
            results.append(result)
            by_key[(smooth_window, degree)] = (rows, baselines)
    if not results:
        print("No trainable candidates.", file=sys.stderr)
        return 2

    results.sort(key=lambda item: (item["test_metrics"]["mae_deg"], item["test_metrics"]["rmse_deg"], item["degree"]))
    selected = results[0]
    rows, baselines = by_key[(selected["smooth_window"], selected["degree"])]
    x_all = np.asarray([row["delta"] for row in rows], dtype=float)
    y_all = np.asarray([row["theta"] for row in rows], dtype=float)
    coeffs = np.polyfit(x_all, y_all, deg=selected["degree"])
    best = {
        "degree": selected["degree"],
        "smooth_window": selected["smooth_window"],
        "coeffs": coeffs,
        "train_metrics": selected["train_metrics"],
        "test_metrics": selected["test_metrics"],
    }
    candidate_summary = [
        {
            "smooth_window": int(item["smooth_window"]),
            "degree": int(item["degree"]),
            "train_mae_deg": float(item["train_metrics"]["mae_deg"]),
            "train_rmse_deg": float(item["train_metrics"]["rmse_deg"]),
            "test_mae_deg": float(item["test_metrics"]["mae_deg"]),
            "test_rmse_deg": float(item["test_metrics"]["rmse_deg"]),
        }
        for item in results[:10]
    ]

    model_path, pred_csv, plot_path = save_outputs(
        args.output_dir,
        args,
        datasets,
        rows,
        baselines,
        best,
        candidate_summary,
    )
    final = metrics(y_all, np.polyval(coeffs, x_all))
    print(f"runs={len(datasets)} samples={len(rows)}")
    print(f"selected_smooth_window={best['smooth_window']} selected_degree={best['degree']}")
    print(f"per_run_baselines={[round(float(v), 3) for v in baselines]}")
    print(f"coefficients_high_to_low={[float(c) for c in coeffs]}")
    print(f"train_mae={best['train_metrics']['mae_deg']:.3f} deg train_rmse={best['train_metrics']['rmse_deg']:.3f} deg")
    print(f"test_mae={best['test_metrics']['mae_deg']:.3f} deg test_rmse={best['test_metrics']['rmse_deg']:.3f} deg")
    print(f"final_all_data_mae={final['mae_deg']:.3f} deg final_all_data_rmse={final['rmse_deg']:.3f} deg")
    print("top_candidates:")
    for item in candidate_summary[:5]:
        print(
            "  smooth=%d degree=%d test_mae=%.3f test_rmse=%.3f"
            % (item["smooth_window"], item["degree"], item["test_mae_deg"], item["test_rmse_deg"])
        )
    print(f"Saved model: {model_path}")
    print(f"Saved predictions: {pred_csv}")
    print(f"Saved plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
