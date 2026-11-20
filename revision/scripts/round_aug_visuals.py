"""Read-only presentation of the augmented-fusion scorer's frozen outputs.

No acquisition, estimator, raw reference, model training, deployment or publishing
is performed here. Importing this module has no filesystem side effects.

Dependencies: numpy, matplotlib, Pillow; video additionally ffmpeg. Native CAD
replay additionally pandas, mujoco and the existing round_g1/round_f modules on
PYTHONPATH. --body-manifest provides a pre-rendered alternative.

Scored NPZ: scored/<case>/<source>-<condition>.npz. ``truth`` and ``b`` contain
bending vectors in radians, not (theta, phi). t is time after release in seconds.
Scores are already aggregated by the scorer; this module never averages RMSEs.

Frame manifest (paths relative to its directory):
  {"case": "26112012-left", "condition": "occluded",
   "frames": [{"capture_s":26.0,"release_s":26.2,"rgb":"input/rgb.png",
                "frame_id":"650"}],
   "phases": {"blind":[31.2,35.2],"degraded":[29.0,31.2]}}
The RGB files MUST be the actual condition-specific input to visual tracking.
Never manufacture a black frame here. A frames.txt can instead be supplied with
--frames-txt and the explicitly declared --capture-delay 0.20.
Body manifest: {"case":..., "frames":[{"capture_s":26.,"rgb":"cad.png"}]}.
Native replay takes --flight-root pointing to the ORIGINAL same-case flight.

Example (generation must be explicitly invoked by the parent task):
  python round_aug_visuals.py --root revision/data/round_augmented_fusion \
      --frame-manifest frames.json --flight-root ../round_strain_health/runs/26112012-left \
      --main --matrix --video
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

METHODS = ("vision", "strain", "fixed", "proposed")
NAMES = ("Vision only", "Strain only", "Fixed-noise KF", "Proposed")
CN_NAMES = ("纯视觉", "纯应变", "固定噪声 KF", "本文方法")
COLORS = ("#5684A0", "#B58B50", "#8D829F", "#398777")
STYLES = ("--", "-.", ":", "-")
SOURCES = ("nominal", "common", "differential", "single", "late")
CONDITIONS = ("natural", "graded", "occluded")
SOURCE_CN = {"nominal": "基础迟滞", "common": "共模漂移", "differential": "差模漂移",
             "single": "单通道偏移", "late": "后期漂移"}
COND_CN = {"natural": "正常视野", "graded": "渐进退化", "occluded": "完全遮挡"}
DEFAULT_EPOCHS = (26.0, 30.6, 33.0, 38.0)  # exposure, selected before looking at scores
INK, MUTED, RULE = "#263A40", "#617176", "#DCE3E3"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_trace(root, case, source, condition):
    if source not in SOURCES or condition not in CONDITIONS:
        raise ValueError("Unknown source or visual condition")
    path = Path(root) / "scored" / case / f"{source}-{condition}.npz"
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k].copy() for k in ("t", "truth", "b", "bias", "gain_v", "gain_s",
                                       "confidence_v", "confidence_s")}
    t = out["t"]
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or not (np.diff(t) > 0).all():
        raise ValueError("Scored release time must be finite and strictly increasing")
    n = len(t)
    for k, shape in {"truth": (n, 2), "b": (4, n, 2), "bias": (n, 2),
                     **{q: (n,) for q in ("gain_v", "gain_s", "confidence_v", "confidence_s")}}.items():
        if out[k].shape != shape:
            raise ValueError(f"{path}: {k} shape {out[k].shape}, expected {shape}")
    if not np.isfinite(out["truth"]).all():
        raise ValueError("Scorer-exported reference contains missing values")
    # Missing vision is intentionally retained as NaN; no interpolation or hold.
    keep=(t>=14.)&(t<46.)
    for k in list(out):
        out[k]=out[k][:,keep] if k=="b" else out[k][keep]
    out["path"] = path
    return out


def angles(v):
    v = np.asarray(v)
    theta = np.linalg.norm(v, axis=-1)
    phi = np.arctan2(v[..., 1], v[..., 0])
    phi = np.where(np.isfinite(v).all(axis=-1) & (theta > 1e-10), phi, np.nan)
    return np.degrees(theta), np.degrees(phi)


def circular_error(pred, truth):
    return np.degrees(np.arctan2(np.sin(np.radians(pred - truth)),
                                np.cos(np.radians(pred - truth))))


def read_scores(root):
    data = read_json(Path(root) / "scores.json")
    if not isinstance(data.get("cases"), dict):
        raise ValueError("scores.json must contain cases[source/condition/phase]")
    unit = data.get("angle_unit", "deg")
    if unit not in ("deg", "degree", "degrees"):
        raise ValueError(f"Expected scorer RMSE in degrees, got {unit}")
    for key, methods in data["cases"].items():
        for method in METHODS:
            r = methods.get(method)
            if r is None:
                continue
            n, valid, phi_n = (int(r[q]) for q in ("n", "valid", "phi_n"))
            if not (0 <= phi_n <= valid <= n):
                raise ValueError(f"Invalid scorer counts: {key}/{method}")
            coverage = r.get("coverage")
            if coverage is not None and not (0 <= float(coverage) <= 1):
                raise ValueError(f"Invalid coverage: {key}/{method}")
    return data


def score_entry(scores, source, condition, phase, method):
    return scores.get("cases", {}).get(f"{source}/{condition}/{phase}", {}).get(method)


def score_value(entry, key):
    if entry is None or entry.get(key) is None:
        return None
    x = float(entry[key])
    return x if math.isfinite(x) else None


class FrameSequence:
    """The actual RGB input sequence, selected causally on RELEASE time."""
    def __init__(self, case, condition, manifest=None, frames_txt=None, delay=None):
        self.case, self.condition, self.phases = case, condition, {}
        self.source = Path(manifest or frames_txt).resolve()
        if manifest:
            doc = read_json(manifest)
            if doc.get("case") != case or doc.get("condition") != condition:
                raise ValueError("Frame manifest case/condition does not match the scored trace")
            self.phases = doc.get("phases", {})
            records = doc["frames"]
        else:
            if delay is None:
                raise ValueError("frames.txt requires explicit capture-to-release delay")
            records = []
            for line in self.source.read_text(encoding="utf-8").splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                fields = line.split()
                capture = float(fields[0])
                records.append(dict(capture_s=capture, release_s=capture + delay,
                                    rgb=fields[1], frame_id=str(len(records))))
        self.rows = []
        for row in records:
            p = Path(row["rgb"])
            p = p if p.is_absolute() else self.source.parent / p
            if not p.is_file():
                raise FileNotFoundError(p)
            self.rows.append(dict(row, rgb=p.resolve()))
        self.capture = np.asarray([float(x["capture_s"]) for x in self.rows])
        self.release = np.asarray([float(x["release_s"]) for x in self.rows])
        if len(self.rows) < 2 or not (np.diff(self.capture) > 0).all() or not (np.diff(self.release) > 0).all():
            raise ValueError("Frame capture/release stamps must be strictly increasing")
        if not (self.release >= self.capture).all():
            raise ValueError("A frame cannot be released before exposure")

    def at_capture(self, stamp, tolerance=.021):
        k = int(np.argmin(abs(self.capture - stamp)))
        if abs(self.capture[k] - stamp) > tolerance:
            raise ValueError(f"No actual frame within {tolerance}s of exposure {stamp}")
        return self.rows[k]

    def at_release(self, stamp):
        k = int(np.searchsorted(self.release, stamp + 1e-8, side="right") - 1)
        if k < 0:
            raise ValueError("Video begins before the first released visual frame")
        return self.rows[k]

    @staticmethod
    def image(row):
        from PIL import Image
        with Image.open(row["rgb"]) as im:
            return im.convert("RGB")


def vision_frames(case, condition, *, manifest=None, frames_txt=None, capture_delay=None):
    """Shared entry point used by both the evidence plate and video."""
    return FrameSequence(case, condition, manifest, frames_txt, capture_delay)


class CachedBody:
    def __init__(self, path, case):
        self.path = Path(path)
        d = read_json(path)
        if d.get("case") != case:
            raise ValueError("Native body images and scorer must use the same case")
        self.rows = d["frames"]
        self.t = np.asarray([float(x["capture_s"]) for x in self.rows])

    def image(self, capture):
        from PIL import Image
        k = int(np.argmin(abs(self.t - capture)))
        if abs(self.t[k] - capture) > .021:
            raise ValueError("No synchronized native CAD frame; do not substitute another flight")
        p = Path(self.rows[k]["rgb"])
        with Image.open(p if p.is_absolute() else self.path.parent / p) as im:
            return im.convert("RGB")

    def close(self):
        pass


class NativeBody:
    """Replay the recorded physical qpos through the existing native CAD camera.

    Only run.json, physics.npz and motor telemetry are used. In particular this
    adapter never opens strain/reference.npz, mocap_b or an old score file.
    The imported Camera applies a rendering-only gate cutaway, as in prior work.
    """
    def __init__(self, folder, case):
        import pandas as pd
        from round_g1_model import make_plant
        from round_g1_visuals import Camera
        from round_f_rotor_replay import split_model, TURNS_PER_SIM_SECOND
        folder = Path(folder)
        if folder.name != case:
            raise ValueError("--flight-root must be the original same-case run directory")
        r = read_json(folder / "run.json")["release_us"]
        with np.load(folder / "physics.npz", allow_pickle=False) as z:
            self.q, self.t = z["qpos"].copy(), (z["us"].copy() - r) / 1e6
        h = pd.read_csv(folder / "hil.csv", usecols=["us", "m1", "m2", "m3", "m4"])
        self.mt = (h.us.to_numpy() - r) / 1e6
        motor = h[["m1", "m2", "m3", "m4"]].to_numpy()
        if np.any(motor < 0):
            raise ValueError("Negative motor telemetry cannot be used as rotor speed")
        steps = np.sqrt(motor[:-1]) * [1, 1, -1, -1] * (np.diff(self.mt) * 2 * np.pi * TURNS_PER_SIM_SECOND)[:, None]
        self.spin = np.vstack([np.zeros(4), np.cumsum(steps, axis=0)])
        self.tmp = tempfile.TemporaryDirectory(prefix="aug-cad-replay-")
        m, axes, _ = split_model(make_plant(), Path(self.tmp.name))
        self.camera = Camera(m, axes)
        self.camera.cam.azimuth, self.camera.cam.elevation = 109., -14.

    def image(self, capture):
        if capture < self.t[0] or capture > self.t[-1]:
            raise ValueError("CAD replay requested outside the recorded flight")
        i, j = int(np.argmin(abs(self.t - capture))), int(np.argmin(abs(self.mt - capture)))
        q = self.q[i]
        return self.camera.image(q, self.spin[j], q[:3] + [.03, .0, -.17], .91)

    def close(self):
        self.camera.ren.close()
        self.tmp.cleanup()


def setup_plot(font_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if font_path:
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "font.size": 8.2, "axes.titlesize": 9,
        "axes.labelsize": 8.2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "axes.edgecolor": "#A7B4B5", "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.fontsize": 7.3, "pdf.fonttype": 42, "svg.fonttype": "none"})
    return plt


def save_plot(fig, out, stem):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=240, facecolor="white")


def decorate(ax, t, phases, markers=()):
    ax.set_xlim(float(t[0]), float(t[-1]))
    for key, color in (("degraded", "#F4EDDF"), ("blind", "#E7EBEE"), ("recovered", "#EEF4F0")):
        if key in phases:
            lo, hi = phases[key]
            ax.axvspan(lo, hi, color=color, zorder=-5)
    for x in markers:
        ax.axvline(x, lw=.55, color="#ABB7B8", ls=":", zorder=-3)
    ax.grid(axis="y", color=RULE, lw=.45, zorder=-4)


def main_plate(trace, frames, body, out, epochs=DEFAULT_EPOCHS, source="differential"):
    import matplotlib.pyplot as plt
    t = trace["t"]
    rows = [frames.at_capture(e) for e in epochs]
    releases = [float(r["release_s"]) for r in rows]
    fig = plt.figure(figsize=(8.0, 8.15))
    gs = fig.add_gridspec(7, 4, left=.080, right=.982, top=.955, bottom=.085,
                         height_ratios=[1.3, .96, .85, .85, .85, 1.07, 1.02], hspace=.64, wspace=.21)
    from scipy.ndimage import binary_dilation,find_objects,label as components
    stages=('Full view','Partial view','Occluded','Relocked')
    for j, row in enumerate(rows):
        im=np.asarray(body.image(float(row['capture_s'])))
        labels,count=components(binary_dilation(im[:,:,:3].min(axis=2)<145,iterations=3))
        if count:
            yy,xx=find_objects(labels)[np.argmax(np.bincount(labels.ravel())[1:])]
            im=im[max(0,yy.start-13):yy.stop+13,max(0,xx.start-13):xx.stop+13]
        a = fig.add_subplot(gs[0, j]); a.imshow(im); a.axis("off")
        a.set_title(f"{chr(97+j)}  {stages[j]}", loc="left", fontsize=9.2,fontweight='bold')
        a = fig.add_subplot(gs[1, j]); a.imshow(frames.image(row)); a.set_xticks([]); a.set_yticks([])
        a.set_xlabel(f"Exposure {float(row['capture_s']):.2f} s", fontsize=7.7)
    axes = []
    labels = ["e  Observation confidence", "f  Mean update gain over 0.2 s", "g  Estimated strain bias",
              "h  Bending magnitude", "i  Signed bending error"]
    for i in range(2, 7):
        a = fig.add_subplot(gs[i, :]); decorate(a, t, frames.phases, releases)
        a.set_title(labels[i-2], loc="left", pad=3, fontsize=8.9)
        if i < 6: a.tick_params(labelbottom=False)
        axes.append(a)
    for a, keys in ((axes[0], ("confidence_v", "confidence_s")), (axes[1], ("gain_v", "gain_s"))):
        for k, color, label in zip(keys, COLORS[:2], ("Vision", "Strain")):
            if k.startswith('gain'):
                bins=np.arange(14.,46.,.2);means=[np.mean(trace[k][(t>=b)&(t<b+.2)]) for b in bins]
                a.plot(bins+.1,means,color=color,lw=1.15,label=label)
            else:a.plot(t, trace[k], color=color, lw=1.15, label=label)
        a.set_ylabel("Confidence" if keys[0].startswith("confidence") else "Update gain")
        a.legend(loc="upper right", ncol=2, frameon=False)
    bias = np.degrees(trace["bias"])
    axes[2].plot(t, bias[:, 0], color=COLORS[3], lw=1.2, label=r"$\hat{\delta}_x$")
    axes[2].plot(t, bias[:, 1], color=COLORS[2], lw=1.2, ls="--", label=r"$\hat{\delta}_y$")
    axes[2].set_ylabel("Bias (deg)"); axes[2].legend(loc="upper right", ncol=2, frameon=False)
    truth, _ = angles(trace["truth"])
    for j in range(4):
        pred, _ = angles(trace["b"][j])
        axes[3].plot(t, pred, color=COLORS[j], lw=1.35 if j == 3 else .95, ls=STYLES[j], label=NAMES[j])
        axes[4].plot(t, pred-truth, color=COLORS[j], lw=1.35 if j == 3 else .95, ls=STYLES[j])
    axes[3].plot(t, truth, color="#263238", lw=.85, alpha=.8, label="Virtual MoCap")
    axes[3].set_ylabel(r"$\theta$ (deg)")
    axes[3].legend(loc="lower center", ncol=5, frameon=False, fontsize=6.8)
    axes[4].axhline(0, color="#859394", lw=.7)
    axes[4].set_ylabel("Error (deg)"); axes[4].set_xlabel("Time after release (s)")
    save_plot(fig, out, "fusion_main"); plt.close(fig)
    return rows


def matrix_plot(scores, out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.patches import Rectangle
    from matplotlib.cm import ScalarMappable
    sources = ("nominal", "common", "differential", "single")
    phases = ("all", "blind")
    rows = [(s, c) for s in sources for c in CONDITIONS]
    values = np.full((2, len(rows), len(METHODS)), np.nan)
    for i, phase in enumerate(phases):
        for j, (s, c) in enumerate(rows):
            for k, method in enumerate(METHODS):
                rec = score_entry(scores, s, c, phase, method)
                if rec is None:
                    raise ValueError(f"Missing matrix record for {s}/{c}/{phase}/{method}")
                value = score_value(rec, "theta")
                if rec["valid"]:
                    if value is None or value <= 0:
                        raise ValueError("Expected positive RMSE for the scored panel")
                    values[i, j, k] = value
                elif value is not None:
                    raise ValueError("An absent observation cannot carry an RMSE")
    lo = min(.2, np.nanmin(values)); hi = max(8., np.nanmax(values))
    norm = LogNorm(lo, hi)
    cmap = LinearSegmentedColormap.from_list("bend_rmse",
                ["#f1f6f7", "#c6dce0", "#8bb5bf", "#477e90", "#17445c"])
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.95))
    fig.subplots_adjust(left=.229, right=.985, top=.835, bottom=.16, wspace=.18)
    col_labels = ("Vision", "Strain", "Fixed KF", "Proposed")
    groups = ("Base hysteresis", "Common drift", "Differential drift", "Single channel")
    for i, ax in enumerate(axs):
        ax.set(xlim=(-.5, 3.5), ylim=(12.56, -.50))
        ax.set_axis_off()
        ax.set_title(("a   Full interval   14–46 s", "b   Matched window   31.2–35.2 s")[i],
                     loc="left", fontsize=11, fontweight="bold", pad=29)
        for k, label in enumerate(col_labels):
            ax.text(k, -.91, label, ha="center", va="center", fontsize=9,
                    color=INK, fontweight="bold" if k==3 else "normal", clip_on=False)
        for j, (_, c) in enumerate(rows):
            y = j + .34*(j//3)
            if i == 0:
                ax.text(-.59, y, {"natural":"Natural","graded":"Graded","occluded":"Occluded"}[c],
                        ha="right", va="center", fontsize=8.5, color=MUTED, clip_on=False)
                if j%3 == 1:
                    ax.text(-1.55, y, groups[j//3], ha="right", va="center",
                            fontsize=9.1, fontweight="bold", clip_on=False)
            for k, value in enumerate(values[i,j]):
                finite = np.isfinite(value)
                background = cmap(norm(value)) if finite else "#eef0f0"
                ax.add_patch(Rectangle((k-.489, y-.484),.978,.968,fc=background,ec="none"))
                label = f"{value:.2f}" if finite else "no pose"
                fg = ("white" if norm(value)>.62 else INK) if finite else MUTED
                ax.text(k, y, label, ha="center", va="center", fontsize=9,
                        color=fg, fontweight="bold" if k==3 and finite else "normal")
    bar_ax = fig.add_axes([.229,.075,.32,.017])
    bar = fig.colorbar(ScalarMappable(norm=norm,cmap=cmap), cax=bar_ax, orientation="horizontal")
    ticks = [.2,.5,1.,2.,4.,8.]
    bar.set_ticks(ticks);bar.set_ticklabels([f"{n:g}" for n in ticks])
    bar.ax.minorticks_off();bar.outline.set_visible(False)
    fig.text(.112,.08,r"$\theta$ RMSE (deg)",fontsize=9,color=INK,va="center")
    fig.text(.60,.08,"Grey cells mark zero visual coverage in the 4 s blackout.",fontsize=8.3,color=MUTED,va="center")
    save_plot(fig, out, "fusion_matrix"); plt.close(fig)


def video(trace, frames, body, out, fps=10., start=None, end=None):
    """Same-case capture imagery with release-time curves; no external truth read."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    t = trace["t"]
    start = max(float(t[0]), float(frames.release[0])) if start is None else start
    end = min(float(t[-1]), float(frames.release[-1])) if end is None else end
    if not (t[0] <= start < end <= t[-1]) or fps <= 0:
        raise ValueError("Invalid video interval or FPS")
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(4, 2, left=.055, right=.985, bottom=.08, top=.91,
                         width_ratios=[1.02, 1.38], hspace=.45, wspace=.24)
    body_ax, wrist_ax = fig.add_subplot(gs[:2, 0]), fig.add_subplot(gs[2:, 0])
    for a in (body_ax, wrist_ax): a.axis("off")
    first = frames.at_release(start)
    body_im = body_ax.imshow(body.image(float(first["capture_s"])))
    wrist_im = wrist_ax.imshow(frames.image(first))
    ax = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    truth, _ = angles(trace["truth"])
    for j in range(4):
        y, _ = angles(trace["b"][j])
        ax[0].plot(t, y, color=COLORS[j], lw=1.4 if j == 3 else 1, ls=STYLES[j], label=NAMES[j])
        ax[1].plot(t, y-truth, color=COLORS[j], lw=1.4 if j == 3 else 1, ls=STYLES[j])
    ax[0].plot(t, truth, color=INK, lw=.8, label="Virtual MoCap")
    for k, col in enumerate(COLORS[:2]):
        ax[2].plot(t, trace[("gain_v", "gain_s")[k]], color=col, lw=1.1, label=("Vision", "Strain")[k])
        ax[3].plot(t, np.degrees(trace["bias"][:, k]), color=(COLORS[3], COLORS[2])[k],
                   lw=1.1, ls=("-", "--")[k], label=("bias x", "bias y")[k])
    cursors = []
    for i, a in enumerate(ax):
        decorate(a, t, frames.phases); a.set_xlim(start, end)
        a.set_ylabel(("Bend (deg)", "Error (deg)", "Update gain", "Bias (deg)")[i])
        cursors.append(a.axvline(start, color="#A6B2B4", lw=.9))
        if i < 3: a.tick_params(labelbottom=False)
    ax[0].legend(loc="upper left", ncol=3, fontsize=6.6, frameon=False)
    ax[2].legend(loc="upper right", ncol=2, fontsize=7, frameon=False)
    ax[3].legend(loc="upper right", ncol=2, fontsize=7, frameon=False)
    ax[3].set_xlabel("Time after release (s)")
    title = fig.text(.055, .965, "", fontsize=12, weight="bold")
    fig.text(.055, .025, "Original-flight native CAD and actual condition-specific visual input; plotted estimators use the release clock.", fontsize=8, color=MUTED)
    dest = Path(out) / "fusion_replay.mp4"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1280x720", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)], stdin=subprocess.PIPE)
    previous = None
    try:
        for now in np.arange(start, end + 1e-9, 1/fps):
            row = frames.at_release(float(now))
            key = str(row["rgb"])
            if key != previous:
                body_im.set_data(body.image(float(row["capture_s"]))); wrist_im.set_data(frames.image(row)); previous = key
            body_ax.set_title(f"Native CAD / exposure {row['capture_s']:.2f} s", loc="left", fontsize=9)
            wrist_ax.set_title(f"Actual visual input / {frames.condition}", loc="left", fontsize=9)
            title.set_text(f"{frames.case}  |  release clock {now:05.2f} s")
            for cursor in cursors: cursor.set_xdata([now, now])
            canvas.draw(); rgb = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
            enc.stdin.write(rgb.tobytes())
    except Exception:
        enc.kill(); raise
    finally:
        enc.stdin.close(); code = enc.wait(); plt.close(fig)
    if code:
        raise RuntimeError(f"ffmpeg failed with code {code}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--case", default="26112012-left")
    p.add_argument("--source", choices=SOURCES, default="differential")
    p.add_argument("--condition", choices=CONDITIONS, default="occluded")
    p.add_argument("--frame-manifest", type=Path); p.add_argument("--frames-txt", type=Path)
    p.add_argument("--capture-delay", type=float, default=.20, help="Registered exposure-to-release delay, seconds (0.20)")
    p.add_argument("--flight-root", type=Path); p.add_argument("--body-manifest", type=Path)
    p.add_argument("--phases", type=Path, help="Optional release-clock phase bounds JSON")
    p.add_argument("--font", type=Path); p.add_argument("--epochs", type=float, nargs=4, default=DEFAULT_EPOCHS)
    p.add_argument("--main", action="store_true"); p.add_argument("--matrix", action="store_true")
    p.add_argument("--video", action="store_true"); p.add_argument("--fps", type=float, default=10)
    p.add_argument("--start", type=float); p.add_argument("--end", type=float)
    a = p.parse_args()
    if not (a.main or a.matrix or a.video): p.error("Select --main, --matrix and/or --video")
    out = a.out or a.root / "review"; out.mkdir(parents=True, exist_ok=True)
    setup_plot(a.font)
    inputs, epoch_rows = [], []
    if a.matrix:
        matrix_plot(read_scores(a.root), out); inputs.append(a.root / "scores.json")
        if not a.main and not a.video and (out/"visual_manifest.json").exists():
            manifest = read_json(out/"visual_manifest.json")
            if sha256(inputs[0]) not in {x["sha256"] for x in manifest["inputs"]}:
                raise ValueError("Matrix scores differ from the existing synchronized plate")
            for item in manifest["outputs"]:
                name = Path(item["path"]).name
                path = out/name
                if name in ("fusion_matrix.png", "fusion_matrix.svg"):
                    item["sha256"] = sha256(path)
                elif sha256(path) != item["sha256"]:
                    raise ValueError("Existing visual changed outside matrix rendering: "+name)
            (out/"visual_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            return
    if a.main or a.video:
        if not a.frame_manifest and not a.frames_txt:
            a.frames_txt = a.root / "vision_inputs" / a.case / a.condition / "frames.txt"
        if bool(a.frame_manifest) == bool(a.frames_txt): p.error("Supply exactly one actual-input frame manifest or frames.txt")
        if bool(a.flight_root) == bool(a.body_manifest): p.error("Supply exactly one same-case native flight or body manifest")
        trace = load_trace(a.root, a.case, a.source, a.condition); inputs.append(trace["path"])
        frames = vision_frames(a.case, a.condition, manifest=a.frame_manifest, frames_txt=a.frames_txt, capture_delay=a.capture_delay)
        inputs.append(frames.source)
        if a.phases: frames.phases = read_json(a.phases); inputs.append(a.phases)
        body = CachedBody(a.body_manifest, a.case) if a.body_manifest else NativeBody(a.flight_root, a.case)
        setup_plot(a.font)  # Legacy camera imports must not change this plate's style.
        try:
            if a.main: epoch_rows = main_plate(trace, frames, body, out, a.epochs, a.source)
            if a.video: video(trace, frames, body, out, a.fps, a.start, a.end)
        finally: body.close()
    manifest = {"case": a.case, "source": a.source, "condition": a.condition,
        "timebase": "score t is release time; CAD/wrist pairs use the same capture time",
        "gains": "per-source per-update; not complementary normalized weights",
        "inputs": [{"path": str(x.resolve()), "sha256": sha256(x)} for x in inputs],
        "epochs": [{**r, "rgb": str(r["rgb"]), "sha256": sha256(r["rgb"])} for r in epoch_rows],
        "truth_access": "only scorer-exported truth in this round's scored NPZ",
        "outputs": [{"path": str(x.resolve()), "sha256": sha256(x)} for x in out.glob("fusion_*") if x.is_file()]}
    (out / "visual_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
