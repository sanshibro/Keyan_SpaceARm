"""Immutable gate records and pose-only CAD access for the M0 publication plates."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

import mujoco as mj
import numpy as np

ROOT = Path(os.environ.get("M0_OUTPUT", Path(__file__).resolve().parents[1]))
_default = ROOT.parent / "冻结感知_单门双自由度精细作业闭环_20261120"
if not _default.is_dir() and (ROOT.parent/"data/round_gate_feedback").is_dir():
    _default=ROOT.parent/"data/round_gate_feedback"
GATE = Path(os.environ.get("M0_GATE_ARCHIVE", _default))
OUT = ROOT / "review"
WORK = ROOT / "work"
QA = ROOT / "qa"
for folder in (OUT, WORK, QA):
    folder.mkdir(parents=True, exist_ok=True)
for source in (GATE/"source", ROOT.parent/"scripts"):
    if source.is_dir():sys.path.insert(0,str(source))
from gate_portable_model import make_plant
from gatereview_visuals import angle, WristFrames

METHODS = ("vision", "strain", "fixed", "proposed", "oracle")
SEED = 26112131
LABELS = dict(vision="Vision only", strain="Strain only", fixed="Fixed KF",
              proposed="Proposed", oracle="Oracle")
COLORS = dict(vision="#a8b8c2", strain="#b16c39", fixed="#816a9c",
              proposed="#137566", oracle="#58666a")


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, obj):
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def protocol():
    for path in (GATE/"protocol.json", GATE/"formal-design.json"):
        if path.is_file():
            obj = read_json(path)
            if "target_axis_enu" in obj:
                return obj
    with zipfile.ZipFile(next((GATE/"evidence").glob("*.zip"))) as z:
        return json.loads(z.read("protocol.json"))


def index():
    with (GATE/"review/evidence-index.csv").open(encoding="utf-8-sig") as f:
        return {r["case"]: r for r in csv.DictReader(f)}


class Run:
    def __init__(self, case, fields=("physics", "control", "score-trace"), verify_zip=True):
        name = GATE / "evidence" / (case + ".zip")
        self.case = case
        if verify_zip and sha_file(name) != index()[case]["sha256"]:
            raise ValueError("Raw archive fingerprint differs: " + case)
        self.inputs = {}
        with zipfile.ZipFile(name) as z:
            locked = json.loads(z.read("evidence-hashes.json"))
            for basename in ("run.json", "control-summary.json", "metrics.json", "protocol.json",
                             *(field+".npz" for field in fields)):
                buf = z.read(basename)
                digest = sha(buf)
                entry = locked[basename]
                expected = entry["sha256"] if isinstance(entry, dict) else entry
                if digest != expected:
                    raise ValueError(f"{case}: {basename} fingerprint differs")
                self.inputs[basename] = digest
                if basename.endswith(".json"):
                    obj = json.loads(buf)
                else:
                    with np.load(io.BytesIO(buf), allow_pickle=False) as f:
                        obj = {k:f[k] for k in f.files}
                key = basename.split(".")[0].replace("-", "_")
                setattr(self, key, obj)
        if hasattr(self, "physics"):
            self.pt = (self.physics["us"] - self.run["release_us"])/1e6
        if hasattr(self, "score_trace"):
            self.z = self.score_trace
            self.t = self.z["t"]
        self.method = self.control_summary["case"]["method"]

    def row(self, t):
        return int(np.argmin(np.abs(self.t-t)))

    def closest(self):
        return int(np.argmin(self.z["arm_tool_clearance"]))

    def pose(self, t):
        return self.physics["qpos"][int(np.argmin(np.abs(self.pt-t)))]

    def original_wrist(self):
        return WristFrames(GATE/"formal"/self.case)


def selected():
    return {m:Run(f"{SEED}-differential-occluded-{m}") for m in METHODS}


def plant():
    candidates = (GATE/"records/registered-cad-assets.tgz", GATE/"registered-cad-assets.tgz")
    asset = next(p for p in candidates if p.is_file())
    os.environ["GATE_REGISTERED_CAD"] = str(asset)
    return make_plant().m


def body_poses(model, state):
    data = mj.MjData(model)
    rotation = np.empty((len(state),model.nbody,3,3))
    xyz = np.empty((len(state),model.nbody,3))
    com = np.empty((len(state),3))
    b = model.body("uav").id
    for k,q in enumerate(state):
        data.qpos[:] = q
        mj.mj_kinematics(model, data)
        rotation[k] = data.xmat.reshape(-1,3,3)
        xyz[k] = data.xpos
        com[k] = data.xipos[b]
    return rotation, xyz, com


def quat_mat(q):
    buf = np.empty(9)
    mj.mju_quat2Mat(buf, np.asarray(q, dtype=float))
    return buf.reshape(3,3)


def unit(x):
    a = np.asarray(x, dtype=float)
    return a/np.linalg.norm(a)


def basis(axis):
    z = unit(axis)
    x = unit(np.cross([0,0,1], z)) if abs(z[2]) < .97 else np.array([1.,0.,0.])
    y = unit(np.cross(z,x))
    x = np.cross(y,z)
    return np.column_stack([x,y,z])


def matrix(R, t):
    H = np.eye(4)
    H[:3,:3] = R
    H[:3,3] = t
    return H
