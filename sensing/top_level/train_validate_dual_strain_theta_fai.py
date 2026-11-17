#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def resolve_csv(path):
    if os.path.isdir(path):
        path = os.path.join(path, "orb_strain_sync.csv")
    return path


def read_run(path, max_theta_age):
    csv_path = resolve_csv(path)
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                age = float(row.get("theta_age") or 999.0)
                valid = int(float(row.get("angle_valid") or 0))
                state = int(float(row.get("tracking_state") or -1))
                if age > max_theta_age or valid != 1 or state != 2:
                    continue
                rows.append(
                    [
                        float(row["rel_time"]),
                        float(row["ch1"]),
                        float(row["ch2"]),
                        float(row["orb_theta_deg"]),
                        float(row["orb_phi_deg"]),
                    ]
                )
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        raise ValueError(f"No usable rows in {csv_path}")
    data = np.asarray(rows, dtype=float)
    zero = data[data[:, 3] < 5.0]
    if len(zero) < 5:
        zero = data[: min(100, len(data))]
    baseline = np.median(zero[:, 1:3], axis=0)
    return csv_path, data, baseline


def feature_names(degree):
    names = []
    for total in range(degree + 1):
        for p1 in range(total, -1, -1):
            p2 = total - p1
            names.append(f"x1^{p1}*x2^{p2}")
    return names


def polynomial_features(x, degree):
    columns = []
    for total in range(degree + 1):
        for p1 in range(total, -1, -1):
            p2 = total - p1
            columns.append((x[:, 0] ** p1) * (x[:, 1] ** p2))
    return np.column_stack(columns)


def fit_ridge(phi, targets, ridge):
    scale = np.std(phi, axis=0)
    scale[scale < 1e-12] = 1.0
    scale[0] = 1.0
    phi_scaled = phi / scale
    reg = np.eye(phi.shape[1]) * ridge
    reg[0, 0] = 0.0
    weights_scaled = np.linalg.solve(phi_scaled.T @ phi_scaled + reg, phi_scaled.T @ targets)
    return weights_scaled / scale[:, None]


def predict_angles(weights, x, degree):
    bend = polynomial_features(x, degree) @ weights
    theta = np.sqrt(np.sum(bend ** 2, axis=1))
    fai = np.degrees(np.arctan2(bend[:, 1], bend[:, 0]))
    fai = np.clip(fai, 0.0, 90.0)
    return theta, fai, bend


def metrics(true, pred):
    err = pred - true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs": float(np.max(np.abs(err))),
    }


def prepare(path, args):
    csv_path, data, baseline = read_run(path, args.max_theta_age)
    mask = (
        (data[:, 3] >= args.min_theta)
        & (data[:, 3] <= args.max_theta)
        & (data[:, 4] >= args.min_fai)
        & (data[:, 4] <= args.max_fai)
    )
    selected = data[mask]
    if len(selected) < 20:
        raise ValueError(f"Not enough target-range rows in {csv_path}: {len(selected)}")
    x = selected[:, 1:3] - baseline
    theta = selected[:, 3]
    fai = selected[:, 4]
    radians = np.radians(fai)
    targets = np.column_stack([theta * np.cos(radians), theta * np.sin(radians)])
    return {
        "csv_path": csv_path,
        "data": selected,
        "baseline": baseline,
        "x": x,
        "theta": theta,
        "fai": fai,
        "targets": targets,
    }


def save_predictions(path, dataset, theta_pred, fai_pred):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rel_time",
                "ch1",
                "ch2",
                "baseline_ch1",
                "baseline_ch2",
                "theta_true_deg",
                "theta_pred_deg",
                "theta_error_deg",
                "fai_true_deg",
                "fai_pred_deg",
                "fai_error_deg",
            ]
        )
        for row, tp, fp in zip(dataset["data"], theta_pred, fai_pred):
            writer.writerow(
                [
                    f"{row[0]:.6f}",
                    f"{row[1]:.6f}",
                    f"{row[2]:.6f}",
                    f"{dataset['baseline'][0]:.6f}",
                    f"{dataset['baseline'][1]:.6f}",
                    f"{row[3]:.6f}",
                    f"{tp:.6f}",
                    f"{tp - row[3]:.6f}",
                    f"{row[4]:.6f}",
                    f"{fp:.6f}",
                    f"{fp - row[4]:.6f}",
                ]
            )


def save_plot(path, dataset, theta_pred, fai_pred, title):
    t = dataset["data"][:, 0]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 9), constrained_layout=True)
    axes[0].plot(t, dataset["theta"], "k.", markersize=2, label="ORB theta")
    axes[0].plot(t, theta_pred, "tab:red", linewidth=1.1, label="pred theta")
    axes[1].plot(t, dataset["fai"], "k.", markersize=2, label="ORB fai")
    axes[1].plot(t, fai_pred, "tab:blue", linewidth=1.1, label="pred fai")
    axes[2].plot(t, theta_pred - dataset["theta"], "tab:red", linewidth=0.9, label="theta error")
    axes[2].plot(t, fai_pred - dataset["fai"], "tab:blue", linewidth=0.9, label="fai error")
    axes[0].set_ylabel("theta (deg)")
    axes[1].set_ylabel("fai (deg)")
    axes[2].set_ylabel("error (deg)")
    axes[2].set_xlabel("time (s)")
    axes[0].set_title(title)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train and validate dual-strain theta/fai model.")
    parser.add_argument("train_runs", nargs="+")
    parser.add_argument("--validation-run", required=True)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--min-theta", type=float, default=10.0)
    parser.add_argument("--max-theta", type=float, default=75.0)
    parser.add_argument("--min-fai", type=float, default=0.0)
    parser.add_argument("--max-fai", type=float, default=90.0)
    parser.add_argument("--max-theta-age", type=float, default=0.2)
    parser.add_argument("--output-dir", default="dual_strain_theta_fai_model")
    args = parser.parse_args()

    train_sets = [prepare(path, args) for path in args.train_runs]
    validation = prepare(args.validation_run, args)
    x_train = np.vstack([ds["x"] for ds in train_sets])
    y_train = np.vstack([ds["targets"] for ds in train_sets])
    phi = polynomial_features(x_train, args.degree)
    weights = fit_ridge(phi, y_train, args.ridge)

    os.makedirs(args.output_dir, exist_ok=True)
    train_theta, train_fai, _ = predict_angles(weights, x_train, args.degree)
    true_train_theta = np.concatenate([ds["theta"] for ds in train_sets])
    true_train_fai = np.concatenate([ds["fai"] for ds in train_sets])
    val_theta, val_fai, _ = predict_angles(weights, validation["x"], args.degree)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": "ridge_polynomial_bend_components",
        "degree": args.degree,
        "ridge": args.ridge,
        "feature_names": feature_names(args.degree),
        "weights_bx_by": weights.tolist(),
        "train_runs": [os.path.abspath(ds["csv_path"]) for ds in train_sets],
        "training_baselines": [ds["baseline"].tolist() for ds in train_sets],
        "validation_run": os.path.abspath(validation["csv_path"]),
        "validation_baseline": validation["baseline"].tolist(),
        "target_range": {
            "theta": [args.min_theta, args.max_theta],
            "fai": [args.min_fai, args.max_fai],
        },
        "train_samples": int(len(x_train)),
        "validation_samples": int(len(validation["x"])),
        "train_theta_metrics_deg": metrics(true_train_theta, train_theta),
        "train_fai_metrics_deg": metrics(true_train_fai, train_fai),
        "validation_theta_metrics_deg": metrics(validation["theta"], val_theta),
        "validation_fai_metrics_deg": metrics(validation["fai"], val_fai),
    }
    model_path = os.path.join(args.output_dir, "dual_strain_theta_fai_model.json")
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    pred_path = os.path.join(args.output_dir, "validation_predictions.csv")
    plot_path = os.path.join(args.output_dir, "validation.png")
    save_predictions(pred_path, validation, val_theta, val_fai)
    save_plot(plot_path, validation, val_theta, val_fai, "Dual strain theta/fai validation")

    print(f"train_samples={len(x_train)} validation_samples={len(validation['x'])}")
    print(f"train_theta_mae={report['train_theta_metrics_deg']['mae']:.3f} deg")
    print(f"train_fai_mae={report['train_fai_metrics_deg']['mae']:.3f} deg")
    print(f"validation_theta_mae={report['validation_theta_metrics_deg']['mae']:.3f} deg")
    print(f"validation_theta_rmse={report['validation_theta_metrics_deg']['rmse']:.3f} deg")
    print(f"validation_fai_mae={report['validation_fai_metrics_deg']['mae']:.3f} deg")
    print(f"validation_fai_rmse={report['validation_fai_metrics_deg']['rmse']:.3f} deg")
    print(f"Saved model/report: {model_path}")
    print(f"Saved predictions: {pred_path}")
    print(f"Saved plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
