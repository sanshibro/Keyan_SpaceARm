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
    x = []
    y = []
    t = []
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
                strain = float(row["strain_value"])
                theta = float(row["orb_theta_deg"])
                rel_time = float(row["rel_time"])
            except (KeyError, TypeError, ValueError):
                continue
            x.append(strain)
            y.append(theta)
            t.append(rel_time)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(t, dtype=float)


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


def train_test_split_by_time(x, y, t, test_fraction):
    order = np.argsort(t)
    x = x[order]
    y = y[order]
    t = t[order]
    n = len(x)
    test_n = max(1, int(round(n * test_fraction)))
    train_n = n - test_n
    if train_n < 2:
        raise ValueError("not enough samples for train/test split")
    return x[:train_n], y[:train_n], t[:train_n], x[train_n:], y[train_n:], t[train_n:]


def metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {"mae_deg": mae, "rmse_deg": rmse, "r2": r2}


def fit_poly(x, y, degree):
    return np.polyfit(x, y, deg=degree)


def evaluate_candidate(x, y, t, degree, smooth_window, test_fraction):
    x_smooth = causal_moving_average(x, smooth_window)
    x_train, y_train, _, x_test, y_test, _ = train_test_split_by_time(x_smooth, y, t, test_fraction)
    baseline = float(np.median(x_train[: max(5, min(200, len(x_train)))]))
    coeffs = fit_poly(x_train - baseline, y_train, degree)
    train_metrics = metrics(y_train, np.polyval(coeffs, x_train - baseline))
    test_metrics = metrics(y_test, np.polyval(coeffs, x_test - baseline))
    return {
        "degree": degree,
        "smooth_window": smooth_window,
        "baseline": baseline,
        "coeffs": coeffs,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


def save_outputs(
    out_dir,
    csv_path,
    channel,
    degree,
    smooth_window,
    baseline,
    coeffs,
    x_raw,
    x_model,
    y,
    t,
    split_index,
    train_metrics,
    test_metrics,
    candidates,
):
    os.makedirs(out_dir, exist_ok=True)
    model = {
        "type": "numpy_polyfit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_csv": os.path.abspath(csv_path),
        "channel": channel,
        "degree": degree,
        "smooth_window": smooth_window,
        "baseline": float(baseline),
        "feature": "causal_moving_average(strain_value, smooth_window) - baseline",
        "coefficients_high_to_low": [float(c) for c in coeffs],
        "prediction": "theta_deg = polyval(coefficients_high_to_low, smoothed_strain - baseline)",
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "split_index": int(split_index),
        "candidate_summary": candidates,
    }
    model_path = os.path.join(out_dir, "strain_theta_model.json")
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)

    pred = np.polyval(coeffs, x_model - baseline)
    pred_csv = os.path.join(out_dir, "predictions.csv")
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rel_time", "split", "strain_value", "model_strain", "theta_true_deg", "theta_pred_deg", "error_deg"])
        for idx, (ti, xi, xmi, yi, pi) in enumerate(zip(t, x_raw, x_model, y, pred)):
            split = "train" if idx < split_index else "test"
            writer.writerow([f"{ti:.6f}", split, f"{xi:.6f}", f"{xmi:.6f}", f"{yi:.6f}", f"{pi:.6f}", f"{pi - yi:.6f}"])

    plot_path = os.path.join(out_dir, "fit.png")
    fig, (ax_time, ax_fit) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    ax_time.plot(t, y, "k.", markersize=2, label="ORB theta")
    ax_time.plot(t, pred, "tab:red", linewidth=1.2, label="predicted theta")
    ax_time.set_xlabel("time (s)")
    ax_time.set_ylabel("theta (deg)")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc="upper left")

    order = np.argsort(x_model)
    ax_fit.plot(x_model, y, "k.", markersize=2, alpha=0.5, label="samples")
    ax_fit.plot(x_model[order], pred[order], "tab:red", linewidth=1.5, label=f"degree {degree} fit")
    ax_fit.set_xlabel("model strain value")
    ax_fit.set_ylabel("theta (deg)")
    ax_fit.grid(True, alpha=0.3)
    ax_fit.legend(loc="upper left")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return model_path, pred_csv, plot_path


def main():
    parser = argparse.ArgumentParser(description="Train strain-to-theta polynomial model.")
    parser.add_argument("csv_or_run_dir", help="orb_strain_sync.csv or its run directory")
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--smooth-window", type=int, default=None)
    parser.add_argument("--no-auto", action="store_true", help="disable candidate search; use --degree and --smooth-window")
    parser.add_argument("--max-theta-age", type=float, default=0.2)
    parser.add_argument("--allow-unusable", action="store_true", help="do not require usable==1")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    csv_path = resolve_csv(args.csv_or_run_dir)
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    x, y, t = read_dataset(
        csv_path,
        args.channel,
        args.max_theta_age,
        require_usable=not args.allow_unusable,
    )
    if len(x) < 20:
        print(f"Not enough usable samples: {len(x)}", file=sys.stderr)
        return 2

    if args.no_auto:
        degrees = [2 if args.degree is None else args.degree]
        windows = [1 if args.smooth_window is None else args.smooth_window]
    else:
        degrees = [1, 2, 3, 4, 5] if args.degree is None else [args.degree]
        windows = [1, 3, 5, 9, 15, 25] if args.smooth_window is None else [args.smooth_window]

    results = []
    for smooth_window in windows:
        for degree in degrees:
            if len(x) < max(20, degree + 3):
                continue
            result = evaluate_candidate(x, y, t, degree, smooth_window, args.test_fraction)
            results.append(result)
    if not results:
        print("No trainable candidates.", file=sys.stderr)
        return 2

    results.sort(key=lambda item: (item["test_metrics"]["mae_deg"], item["test_metrics"]["rmse_deg"]))
    best = results[0]
    split_index = len(x) - max(1, int(round(len(x) * args.test_fraction)))

    x_model = causal_moving_average(x, best["smooth_window"])
    baseline = float(np.median(x_model[: max(5, min(200, len(x_model)))]))
    coeffs = fit_poly(x_model - baseline, y, best["degree"])
    final_metrics = metrics(y, np.polyval(coeffs, x_model - baseline))
    train_metrics = best["train_metrics"]
    test_metrics = best["test_metrics"]
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

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(csv_path), "strain_theta_model")

    model_path, pred_csv, plot_path = save_outputs(
        out_dir,
        csv_path,
        args.channel,
        best["degree"],
        best["smooth_window"],
        baseline,
        coeffs,
        x,
        x_model,
        y,
        t,
        split_index,
        train_metrics,
        test_metrics,
        candidate_summary,
    )

    print(f"samples={len(x)} train={split_index} test={len(x) - split_index}")
    print(f"selected_smooth_window={best['smooth_window']} selected_degree={best['degree']}")
    print(f"baseline={baseline:.6f}")
    print(f"coefficients_high_to_low={[float(c) for c in coeffs]}")
    print(f"train_mae={train_metrics['mae_deg']:.3f} deg train_rmse={train_metrics['rmse_deg']:.3f} deg")
    print(f"test_mae={test_metrics['mae_deg']:.3f} deg test_rmse={test_metrics['rmse_deg']:.3f} deg")
    print(f"final_all_data_mae={final_metrics['mae_deg']:.3f} deg final_all_data_rmse={final_metrics['rmse_deg']:.3f} deg")
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
