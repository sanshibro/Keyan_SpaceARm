"""Native C2 geometry replay for round D. No time integration or flight imagery."""

from pathlib import Path
import os, sys, math, csv, subprocess, json

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "mujoco_closed_loop"))
from super_metric_model import build_super_metric
from model import SimArmBridge
from PIL import Image, ImageDraw, ImageFont

out = R / "revision/data/round_d"
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)
teal = "#226c70"
rust = "#a84d33"
ink = "#253b45"


def run():
    xml, info = build_super_metric(fixed=True)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    bridge = SimArmBridge()
    pair = [m.geom(k).id for k in ("arm7", "collision_landing_168")]
    height, width = 620, 648
    renderer = mujoco.Renderer(m, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 1.26]
    cam.distance = 1.1
    cam.azimuth = 24
    cam.elevation = -8
    cam.orthographic = 1
    opt = mujoco.MjvOption()
    opt.geomgroup[0] = 0
    opt.geomgroup[3] = 0
    opt.sitegroup[:] = 0

    def draw(overlay=False):
        opt.geomgroup[3] = int(overlay)
        renderer.update_scene(d, camera=cam, scene_option=opt)
        return renderer.render().copy()

    def pose(theta, phi):
        bridge.set_kinematic_shape(m, d, math.radians(theta), math.radians(phi))
        mujoco.mj_forward(m, d)

    def distance():
        endpoints = np.zeros(6)
        mm = mujoco.mj_geomDistance(m, d, pair[0], pair[1], 1.0, endpoints) * 1000.0
        matched = [c for c in d.contact[: d.ncon] if {c.geom1, c.geom2} == set(pair)]
        if matched:  # MuJoCo 3.2.x witness depths for box/capsule in penetration.
            ct = min(matched, key=lambda x: x.dist)
            mm = float(ct.dist * 1000)
            pt = ct.pos.copy()
        else:
            pt = (endpoints[:3] + endpoints[3:]) / 2
        other = [c for c in d.contact[: d.ncon] if c.dist < -1.0e-6]
        return mm, pt, bool(other)

    def project(p):
        cc = renderer.scene.camera[0]
        forward = np.asarray(cc.forward)
        up = np.asarray(cc.up)
        right = np.cross(forward, up)
        # For a mono image the stereo camera positions are averaged.
        center = (np.asarray(cc.pos) + np.asarray(renderer.scene.camera[1].pos)) / 2
        v = p - center
        h = float(cc.frustum_top - cc.frustum_bottom)
        return (width * 0.5 + height * np.dot(v, right) / h, height * (0.5 - np.dot(v, up) / h))

    rows = []
    shots = {}
    positions = {}
    for theta in np.linspace(0, 90, 181):
        pose(theta, 90)
        dd, pt, hit = distance()
        rows.append((theta, dd, int(hit)))
        if theta in (0, 60, 90):
            shots[int(theta)] = draw()
            positions[int(theta)] = project(pt)
    with (out / "shape_sweep.csv").open("w") as f:
        wr = csv.writer(f)
        wr.writerow(["theta_deg", "arm7_to_landing168_signed_mm", "any_self_penetration"])
        wr.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.45))
    for i, t in enumerate((0, 60, 90)):
        ax = axes[i]
        ax.imshow(shots[t])
        ax.axis("off")
        ax.set_title("(%s)  Bend %d°" % ("abc"[i], t), loc="left", color=ink, fontsize=12)
        v = rows[t * 2][1]
        ax.text(
            0.025,
            0.035,
            "tracked gap: %+.1f mm" % v,
            transform=ax.transAxes,
            color=rust if t == 60 else teal,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=2),
        )
        if t == 60:
            x, y = positions[t]
            ax.add_patch(Circle((x, y), 14, ec=rust, fc="none", lw=1.7))
            ax.annotate(
                "arm / skid support",
                xy=(x + 9, y - 7),
                xytext=(width * 0.94, height * 0.38),
                ha="right",
                fontsize=8.3,
                color=rust,
                arrowprops=dict(arrowstyle="-", color=rust, lw=0.8),
                bbox=dict(facecolor="white", alpha=0.84, edgecolor="none", pad=1),
            )
    fig.subplots_adjust(left=0.02, right=0.992, bottom=0.025, top=0.89, wspace=0.035)
    fig.savefig(out / "model_sweep.png", dpi=195)
    fig.savefig(out / "model_sweep.pdf")
    plt.close(fig)
    ar = np.array(rows)
    fig, ax = plt.subplots(figsize=(7.3, 3.12))
    ax.fill_between(ar[:, 0], ar[:, 1], 0, where=ar[:, 1] < 0, interpolate=True, color="#f1dbd3")
    ax.axhline(0, color=ink, lw=0.9)
    ax.plot(ar[:, 0], ar[:, 1], color=teal, lw=1.7)
    ax.grid(axis="y", lw=0.45, color="#dbe2e4")
    ax.set(
        xlim=(-2, 92),
        xlabel="Bending angle at fixed plane 90° (deg)",
        ylabel="Signed pair distance (mm)",
    )
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    for t in (0, 60, 90):
        v = ar[t * 2, 1]
        ax.plot(t, v, "o", color=rust if v < 0 else teal, ms=5)
        ax.annotate(
            "%+.1f" % v,
            (t, v),
            xytext=((8 if t == 0 else -10), 8),
            textcoords="offset points",
            ha="left" if t == 0 else "right",
            color=ink,
            fontsize=9,
        )
    ax.set_title("Distal arm segment to landing support", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "distance_curve.png", dpi=200)
    fig.savefig(out / "distance_curve.pdf")
    plt.close(fig)
    pose(90, 180)
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.45))
    overlay = m.geom_rgba.copy()
    for gid in range(m.ngeom):
        if m.geom_group[gid] == 3:
            m.geom_rgba[gid] = [0.22, 0.60, 0.64, 0.38]
    for ax, (nm, az, el) in zip(axs, [("Side", 90, -3), ("Top", 90, -89.5)]):
        cam.azimuth = az
        cam.elevation = el
        ax.imshow(draw(True))
        ax.axis("off")
        ax.set_title(nm + " | CAD and collision proxies", loc="left", fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.01, top=0.91, wspace=0.03)
    fig.savefig(out / "model_overlay.png", dpi=200)
    fig.savefig(out / "model_overlay.pdf")
    plt.close(fig)
    m.geom_rgba[:] = overlay
    cam.azimuth = 24
    cam.elevation = -8
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            "18",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(out / "geometry_replay.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    for t in np.linspace(0, 90, 91):
        pose(t, 90)
        v, pt, hit = distance()
        im = Image.fromarray(draw())
        dr = ImageDraw.Draw(im)
        dr.rectangle((0, 0, width, 52), fill="white")
        dr.text((13, 3), "Geometric replay | fixed bending plane: 90 deg", fill=ink, font=font)
        dr.text(
            (13, 27),
            "Bend %3.0f deg    Pair distance %+5.1f mm" % (t, v),
            fill=rust if v < 0 else teal,
            font=font,
        )
        if v < 0:
            x, y = project(pt)
            dr.ellipse((x - 10, y - 10, x + 10, y + 10), outline=rust, width=3)
        ff.stdin.write(im.tobytes())
    ff.stdin.close()
    assert ff.wait() == 0
    renderer.close()
    print("round D geometry replay:", [(t, rows[2 * t]) for t in (0, 60, 90)], flush=True)


if __name__ == "__main__":
    run()
