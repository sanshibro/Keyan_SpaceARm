"""Render the native meshes in a metric side section of the registered gate."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mujoco
import numpy as np
from round_d_acceptance import Independent, state


def gate_figure(run, out):
    ref = Independent()
    model = ref.m
    model.vis.global_.fovy = 1.56
    model.geom_group[list(ref.gate)] = 4
    height, width = 840, 612
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.0, 1.0]
    camera.azimuth, camera.elevation = 90, 0
    camera.distance, camera.orthographic = 2.0, 1
    option = mujoco.MjvOption()
    option.geomgroup[:] = [0, 1, 1, 0, 0, 0]
    option.sitegroup[:] = 0
    states = np.loadtxt(run / "planned_states.csv", delimiter=",", skiprows=1)
    crossing = states[np.argmin(abs(states[:, 1]))][1:]
    review = json.loads((run / "receiver.json").read_text())
    candidates = [
        ("a", "Planned crossing", crossing, None),
        ("b", "Skid at the sill", state(z=0.80, theta=118, phi=180), "landing_"),
        ("c", "Tool at the sill", state(z=0.88), "gripper_finger"),
    ]
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(8.25, 4.45), sharey=True)
    ink, teal, rust = "#243842", "#276d72", "#a94b31"
    for ax, (letter, title, pose, witness_kind) in zip(axes, candidates):
        ref.at(pose)
        renderer.update_scene(ref.d, camera=camera, scene_option=option)
        image = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(ref.d, camera=camera, scene_option=option)
        mask = renderer.render()[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM
        renderer.disable_segmentation_rendering()
        cam0, cam1 = renderer.scene.camera
        origin = (np.asarray(cam0.pos) + np.asarray(cam1.pos)) / 2
        right = np.cross(cam0.forward, cam0.up)
        span = float(cam0.frustum_top - cam0.frustum_bottom)
        if right[0] < 0:
            image, mask = image[:, ::-1], mask[:, ::-1]
        cx, cz = origin[[0, 2]]
        ext = [
            cx - span * width / (2 * height),
            cx + span * width / (2 * height),
            cz - span / 2,
            cz + span / 2,
        ]
        rgba = np.dstack([image, (mask * 255).astype(np.uint8)])
        ax.axvline(0, color="#8d9b9e", lw=0.62, dashes=(3, 3), zorder=0)
        ax.imshow(rgba, extent=ext, interpolation="lanczos", zorder=2)
        ax.add_patch(
            Rectangle(
                (-0.05, 0),
                0.10,
                0.45,
                fc="#bfc8c9",
                ec="#708084",
                lw=0.8,
                hatch="///",
                alpha=0.72,
                zorder=3,
            )
        )
        ax.set(
            xlim=(-0.43, 0.43),
            ylim=(0.27, 1.73),
            xticks=[-0.3, 0, 0.3],
            yticks=[0.45, 0.75, 1.0, 1.25, 1.5],
        )
        for name in ("top", "right"):
            ax.spines[name].set_visible(False)
        ax.tick_params(length=3, width=0.7, colors=ink)
        ax.set_title(f"({letter})  {title}", loc="left", color=ink, fontsize=11, pad=9)
        ax.set_xlabel("Forward position x (m)")
        if witness_kind:
            hits = [
                c
                for c in ref.d.contact[: ref.d.ncon]
                if c.dist < 0
                and (c.geom1 in ref.gate or c.geom2 in ref.gate)
                and any(witness_kind in (model.geom(i).name or "") for i in (c.geom1, c.geom2))
            ]
            hit = min(hits, key=lambda c: c.dist)
            x, _, z = hit.pos
            ax.scatter([x], [z], s=75, facecolors="none", edgecolors=rust, lw=1.2, zorder=5)
            ax.annotate(
                f"{1000*hit.dist:+.1f} mm",
                xy=(x, z),
                xytext=(0.115, 0.33),
                color=rust,
                arrowprops={"arrowstyle": "-", "lw": 0.85, "color": rust},
                zorder=6,
            )
        else:
            ax.text(-0.40, 0.57, "Sill", fontsize=8, color=ink)
            ax.annotate(
                "",
                xy=(-0.047, 0.447),
                xytext=(-0.27, 0.553),
                arrowprops={"arrowstyle": "-", "lw": 0.7, "color": ink},
            )
    axes[0].set_ylabel("World height z (m)")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.145, top=0.91, wspace=0.18)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "gate_candidates.pdf")
    fig.savefig(out / "gate_candidates.png", dpi=220)
    plt.close(fig)
    renderer.close()
    print(
        "Native gate sections; accepted body and arm minima (m):",
        review["proof"]["body_min"],
        review["proof"]["arm_min"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    gate_figure(args.run, args.out)
