"""Scientific presentation of recorded gate-feedback flights (no simulation runs).

Inputs: protocol.json cases=[{case,method,material,vision,seed}], each case under
<root>/<development|formal>/<case>/ with score-trace.npz, physics.npz, run.json,
and optics/index.jsonl (actual tracker input pixels). The score trace is produced
by the independent scorer, never by this presentation code.

score-trace arrays: t; des_b/est_b/true_b/vision_hold_b [N,2] radians;
body_clearance/arm_tool_clearance/self_clearance/tip_error [N] metres;
axis_error_deg; phase; hold/gate_exit/blackout [N]. Optional near_robot/near_gate
[N,3] are world coordinates from the scorer. Each method has its OWN trajectory.

Dependencies: numpy, matplotlib, Pillow. Video: ffmpeg. Native CAD: mujoco and
gate_feedback_model.make_plant() on PYTHONPATH. No old G1 scene is substituted.
Two fixed cameras are supplied in JSON, e.g. {side:{lookat:[...],azimuth:...,
elevation:...,distance:...,orthographic:true},oblique:{...}}. No auto per-method
zoom is used. All paths and model identity are recorded in visual_manifest.json.

Import has no file-writing or simulator side effects. Generating the standalone
PDF figures is an explicit action; the parent handles its PDF operation marker.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

METHODS = ("vision", "strain", "fixed", "proposed", "oracle")
NAMES = {"vision": "Vision only", "strain": "Strain only", "fixed": "Fixed-noise KF",
         "proposed": "Proposed", "oracle": "Oracle reference"}
CN = {"vision": "纯视觉", "strain": "纯应变", "fixed": "固定 KF", "proposed": "本文方法", "oracle": "Oracle"}
COLORS = {"vision": "#608AA2", "strain": "#B3915D", "fixed": "#9585A5", "proposed": "#478B7A", "oracle": "#45515A"}
INK, MUTED, PALE, RED = "#293B42", "#65767A", "#E2E8E8", "#B56959"
CLEARANCE_M, TIP_M, AXIS_DEG, DWELL_S = .06, .035, 3., 1.
VECTORS = ("des_b", "est_b", "true_b", "vision_hold_b")
SERIES = ("body_clearance", "arm_tool_clearance", "self_clearance", "tip_error", "axis_error_deg", "phase", "hold", "gate_exit", "blackout")


def readj(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def angle(v, phi_min_deg=5.):
    v = np.asarray(v); theta = np.degrees(np.linalg.norm(v, axis=-1))
    phi = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    phi = np.where(np.isfinite(v).all(axis=-1) & (theta >= phi_min_deg), phi, np.nan)
    ok=np.isfinite(phi)
    if ok.any():
        shown=np.degrees(np.unwrap(np.radians(phi[ok])))
        shown+=360*round((180.-shown[0])/360)
        phi[ok]=shown  # Preserve gaps; unwrap only chooses a displayed angular branch.
    return theta, phi


def contiguous(t, mask):
    """Half-open intervals from a sample-held mask; never bridge a missing run."""
    t, mask = np.asarray(t), np.asarray(mask, dtype=bool)
    if not len(t): return []
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return [(float(t[a]), float(t[min(b+1, len(t)-1)])) for a,b in zip(starts, ends)]


@dataclass
class Flight:
    folder: Path
    meta: dict
    z: dict

    @property
    def case(self): return str(self.meta["case"])
    @property
    def method(self): return self.meta["method"]
    @property
    def t(self): return self.z["t"]

    def index(self, t):
        if not len(self.t) or t < self.t[0] or t > self.t[-1]+1e-8: return None
        return int(np.searchsorted(self.t, t+1e-8, side="right")-1)


def load_flight(folder, meta):
    folder = Path(folder)
    with np.load(folder/"score-trace.npz", allow_pickle=False) as f:
        z = {k:f[k].copy() for k in ("t", *VECTORS, *SERIES)}
        for k in ("near_robot", "near_gate", "arm_near_robot", "arm_near_gate", "tip", "tool_axis", "dwell_s"):
            if k in f: z[k] = f[k].copy()
    n = len(z["t"])
    if z["t"].shape != (n,) or not np.isfinite(z["t"]).all() or not (np.diff(z["t"])>0).all():
        raise ValueError(f"{folder}: score time must be strictly increasing release seconds")
    for k in VECTORS:
        if z[k].shape != (n,2): raise ValueError(f"{folder}: {k} must be N x 2 bending vectors")
    for k in SERIES:
        if z[k].shape != (n,): raise ValueError(f"{folder}: {k} must be length N")
    for k in ("near_robot", "near_gate"):
        if k in z and z[k].shape != (n,3): raise ValueError(f"{k} must be N x 3 world coordinates")
    if meta["method"] not in METHODS: raise ValueError("Unknown method")
    return Flight(folder.resolve(), dict(meta), z)


def protocol_cases(path):
    d = readj(path)
    rows = d.get("cases", d.get("runs"))
    if isinstance(rows, dict): rows = [dict(v, case=k) for k,v in rows.items()]
    if not isinstance(rows, list): raise ValueError("Protocol requires a cases/runs list with explicit metadata")
    for r in rows:
        for k in ("case", "method", "material", "vision", "seed"):
            if k not in r: raise ValueError(f"Protocol case missing {k}; do not guess it from a filename")
    if len({r["case"] for r in rows}) != len(rows): raise ValueError("Duplicate case identifiers")
    return rows


def select_group(root, split, rows, seed, material, vision, allow_incomplete=False):
    picked = [r for r in rows if str(r["seed"]) == str(seed) and r["material"]==material and r["vision"]==vision]
    result={}
    for r in picked:
        if r["method"] in result: raise ValueError("More than one flight for a method in selected paired group")
        p=Path(root)/split/r["case"]
        if (p/"score-trace.npz").is_file(): result[r["method"]]=load_flight(p,r)
        elif not allow_incomplete: raise FileNotFoundError(p/"score-trace.npz")
    if not allow_incomplete and set(result)!=set(METHODS): raise ValueError("Selected paired group does not contain all five independently recorded methods")
    if not result: raise ValueError("No recorded flights in selected paired group")
    return result


class WristFrames:
    def __init__(self, folder):
        self.folder=Path(folder)/"optics"
        self.rows=[json.loads(x) for x in (self.folder/"index.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        self.capture=np.array([float(r["t"]) for r in self.rows])
        self.arrival=np.array([float(r["arrival_t"]) for r in self.rows])
        if len(self.rows) and (not (np.diff(self.capture)>0).all() or not (np.diff(self.arrival)>=0).all()):
            raise ValueError("Visual frame order is not monotonic")

    def at(self, now):
        k=int(np.searchsorted(self.arrival,now+1e-8,side="right")-1)
        return None if k<0 else self.rows[k]

    def image(self,row):
        from PIL import Image
        p=Path(row["rgb"]);p=p if p.is_absolute() else self.folder/p
        with Image.open(p) as im:return im.convert("RGB")


class NativeViews:
    """Two fixed cameras on the actual new gate plant, without resetting physics."""
    def __init__(self, flights, cameras, model_module="gate_feedback_model", size=(480,360)):
        import mujoco
        self.mj=mujoco;self.size=size
        module=importlib.import_module(model_module)
        self.plant=module.make_plant();self.m=self.plant.m
        # A floor-free world view retains the physical gate in group 4.
        # The observer model is private to this recording renderer.
        self.m.geom_group[self.m.geom("floor").id]=5
        self.d=mujoco.MjData(self.m)
        self.ren=mujoco.Renderer(self.m,width=size[0],height=size[1])
        self.cameras=cameras;self.flights=flights;self.records={};self.targets={}
        for name in ("side","oblique"):
            for key in ("lookat","azimuth","elevation","distance"):
                if key not in cameras[name]:raise ValueError(f"Fixed camera {name} missing {key}")
        for method,run in flights.items():
            spec=readj(run.folder/"control-summary.json")
            self.targets[method]=(np.array(spec["goal_xyz"]),np.array(spec["goal_axis"]))
            release=readj(run.folder/"run.json")["release_us"]
            with np.load(run.folder/"physics.npz",allow_pickle=False) as ph:
                tt=(ph["us"].copy()-release)/1e6;qq=ph["qpos"].copy()
            if qq.shape[1]!=self.m.nq:raise ValueError("Recorded qpos does not match the new gate native model")
            self.records[method]=(tt,qq)

    def image(self, method, stamp, view, points=None, region=None):
        from PIL import Image
        tt,qq=self.records[method]
        if stamp<tt[0] or stamp>tt[-1]:return None, None
        i=int(np.argmin(abs(tt-stamp)));self.d.qpos[:]=qq[i]
        self.mj.mj_forward(self.m,self.d)
        cfg=self.cameras[view];cam=self.mj.MjvCamera()
        cam.lookat[:]=cfg["lookat"] if region is None else region[0]
        cam.azimuth=cfg["azimuth"];cam.elevation=cfg["elevation"]
        cam.distance=cfg["distance"];cam.orthographic=int(cfg.get("orthographic",True))
        # In orthographic mode the projection span is in metres and is independent of distance.
        if cam.orthographic:self.m.vis.global_.fovy=cfg.get("span_m",1.9) if region is None else region[1]
        option=self.mj.MjvOption()
        option.geomgroup[:]=cfg.get("geomgroup",[1,1,1,0,1,0])
        option.geomgroup[5]=0  # Sensor-only texture canvas must not appear in world-view CAD.
        self.ren.update_scene(self.d,camera=cam,scene_option=option)
        im=Image.fromarray(self.ren.render().copy()).convert("RGB")
        projected=None
        if points is not None and cam.orthographic and np.isfinite(points).all():
            left,right=self.ren.scene.camera;eye=(np.asarray(left.pos)+np.asarray(right.pos))/2
            v=np.asarray(points)-eye;span=left.frustum_top-left.frustum_bottom
            # Same world-to-pixel projection as the prior native orthographic viewer.
            h,w=self.size[1],self.size[0]
            projected=np.column_stack((w/2+h*(v@np.cross(left.forward,left.up))/span,
                                       h/2-h*(v@left.up)/span))
        return im,projected

    def close(self):self.ren.close()


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.3,"axes.labelsize":8.3,
        "axes.titlesize":9,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.6,
        "axes.edgecolor":"#AAB6B7","axes.labelcolor":INK,"text.color":INK,
        "xtick.color":MUTED,"ytick.color":MUTED,"pdf.fonttype":42,"svg.fonttype":"none"})
    return plt


def save(fig,out,stem):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for ext in ("png","pdf","svg"):fig.savefig(out/f"{stem}.{ext}",dpi=240,facecolor="white")


def missing(ax,label="No recorded output"):
    ax.set_facecolor("#F3F5F5");ax.text(.5,.5,label,ha="center",va="center",transform=ax.transAxes,color=MUTED,fontsize=8)
    ax.set_xticks([]);ax.set_yticks([])


def phase_barcode(flights,out):
    """Stage colors, blackout stripe, and no-estimate markers use recorded masks."""
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9.2,2.6));fig.subplots_adjust(left=.15,right=.98,bottom=.24,top=.83)
    all_phases=sorted({str(p) for f in flights.values() for p in f.z["phase"]})
    palette=("#D9E4E9","#C2D8E0","#A7C9C5","#D9D0E3","#E6D6B9","#CBDCCB","#DBCDD1","#CAD4DB")
    cmap={p:palette[i%len(palette)] for i,p in enumerate(all_phases)}
    for j,m in enumerate(METHODS):
        f=flights.get(m)
        if f is None:continue
        phases=f.z["phase"].astype(str)
        for p in all_phases:
            for lo,hi in contiguous(f.t,phases==p):ax.broken_barh([(lo,hi-lo)],(j-.30,.50),facecolors=cmap[p],edgecolors="none")
        for key,color,dy in (("blackout","#333F47",.23),("hold",RED,.34)):
            for lo,hi in contiguous(f.t,f.z[key]):ax.plot([lo,hi],[j+dy]*2,color=color,lw=2.2)
        bad=~np.isfinite(f.z["est_b"]).all(axis=1)
        for lo,hi in contiguous(f.t,bad):ax.broken_barh([(lo,hi-lo)],(j-.30,.50),facecolors="none",edgecolors=RED,hatch="////",linewidth=0)
    from matplotlib.patches import Patch
    handles=[Patch(facecolor=cmap[p],label=p) for p in all_phases]
    handles.append(Patch(facecolor="white",edgecolor=RED,hatch="////",label="No estimator output"))
    ax.legend(handles=handles,ncol=min(5,len(handles)),fontsize=6.7,frameon=False,loc="lower left",bbox_to_anchor=(0,1.04))
    ax.set_yticks(range(5),[NAMES[m] for m in METHODS]);ax.set_ylim(4.6,-.6)
    ax.set_xlabel("Time after release (s)");ax.spines[["left","bottom"]].set_visible(False)
    fig.text(.15,.02,"Dark lower stripe: actual black pixels. Rust stripe: recorded protective hold. Each row is an independent closed-loop flight.",fontsize=7.5,color=MUTED)
    save(fig,out,"gate_state_barcode");plt.close(fig)


def detailed_curves(flights,out):
    """Four physical questions x five independent controllers, on a common clock."""
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,5,figsize=(12.4,8.1),sharex=True,sharey="row")
    fig.subplots_adjust(left=.065,right=.963,top=.90,bottom=.105,wspace=.13,hspace=.24)
    max_t=max((f.t[-1] for f in flights.values() if len(f.t)),default=1)
    for j,m in enumerate(METHODS):
        f=flights.get(m);axes[0,j].set_title(NAMES[m],color=COLORS[m],fontsize=9.4,pad=13)
        if f is None:
            for a in axes[:,j]:missing(a)
            continue
        t=f.t;z=f.z;color=COLORS[m]
        for row in (0,1):
            for key,ls,c,lw,label in (("des_b",":",MUTED,1.,"Desired"),("true_b","-",color,1.35,"Physical"),("est_b","--","#353F46",.9,"Estimate")):
                a,b=angle(z[key]);axes[row,j].plot(t,(a,b)[row],ls=ls,color=c,lw=lw,label=label)
            if m=="vision":
                a,b=angle(z["vision_hold_b"]);axes[row,j].plot(t,(a,b)[row],color="#B59E85",ls="-.",lw=.8,label="Vision hold (diagnostic)")
        axes[2,j].plot(t,100*z["body_clearance"],color="#8EAFBF",lw=1,label="Body")
        axes[2,j].plot(t,100*z["arm_tool_clearance"],color=color,lw=1.35,label="Arm + tool")
        axes[2,j].axhline(100*CLEARANCE_M,color=RED,lw=.8,ls="--")
        axes[3,j].plot(t,100*z["tip_error"],color=color,lw=1.3,label="Position (cm)")
        axes[3,j].axhline(100*TIP_M,color=color,lw=.8,ls=":")
        twin=axes[3,j].twinx();twin.plot(t,z["axis_error_deg"],color=RED,lw=.9,label="Axis (deg)")
        twin.axhline(AXIS_DEG,color=RED,lw=.8,ls="--");twin.set_ylim(0,max(10.,max(float(np.nanmax(g.z["axis_error_deg"])) for g in flights.values() if len(g.t))))
        twin.tick_params(axis="y",labelsize=7,colors=RED,labelright=(j==4));twin.spines["right"].set_visible(j==4)
        if j==4:twin.set_ylabel("Axis error (deg)",color=RED,fontsize=8)
        for a in axes[:,j]:
            a.set_xlim(0,max_t);a.grid(axis="y",color=PALE,lw=.45)
            for lo,hi in contiguous(t,z["blackout"]):a.axvspan(lo,hi,color="#E8ECEF",zorder=-5)
            for lo,hi in contiguous(t,z["hold"]):a.axvspan(lo,hi,facecolor="none",edgecolor="#DAC4BD",hatch="...",lw=0,zorder=-4)
        axes[3,j].set_xlabel("Release time (s)",fontsize=7.5)
    for row,label in enumerate((r"$\theta$ (deg)",r"$\phi$ (deg)","External clearance (cm)","Tip position error (cm)")):axes[row,0].set_ylabel(label)
    handles,labels=next((axes[0,j].get_legend_handles_labels() for j,m in enumerate(METHODS) if m in flights))
    fig.legend(handles,labels,ncol=4,loc="upper center",bbox_to_anchor=(.51,.983),frameon=False,fontsize=8)
    fig.text(.065,.045,"Gray: this flight's actual blackout; dotted background: recorded protective hold. No-output estimates remain gaps; φ is omitted below 5° bending.",fontsize=8,color=MUTED)
    fig.text(.065,.022,"Thresholds: external clearance ≥ 6.0 cm; tip position ≤ 3.5 cm and tool-axis error ≤ 3.0° continuously for 1.0 s. Axis error uses the rust right scale.",fontsize=8,color=MUTED)
    save(fig,out,"gate_method_details");plt.close(fig)


def curves(flights,out):
    """Compact four-row comparison; independent trajectories never share truth."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig,axs=plt.subplots(4,1,figsize=(8.2,5.35),sharex=True)
    fig.subplots_adjust(left=.095,right=.91,top=.875,bottom=.12,hspace=.25)
    twin=axs[3].twinx();twin.spines["right"].set_visible(True)
    max_t=max(f.t[-1] for f in flights.values() if len(f.t))
    for m,f in flights.items():
        z=f.z;t=f.t;c=COLORS[m]
        for row in (0,1):
            axs[row].plot(t,angle(z["true_b"])[row],color=c,lw=1.2)
            if m!="oracle":axs[row].plot(t,angle(z["est_b"])[row],color=c,lw=.8,ls="--",alpha=.85)
        axs[2].plot(t,100*z["arm_tool_clearance"],color=c,lw=1.2)
        axs[2].plot(t,100*z["body_clearance"],color=c,lw=.75,ls=":",alpha=.75)
        dock=z["gate_exit"].astype(bool)
        axs[3].plot(t[dock],100*z["tip_error"][dock],color=c,lw=1.15)
        twin.plot(t[dock],z["axis_error_deg"][dock],color=c,lw=.8,ls="--",alpha=.8)
    if "proposed" in flights:
        f=flights["proposed"]
        black=f.z["blackout"].astype(bool)
        for row in (0,1):
            axs[row].plot(f.t,angle(f.z["des_b"])[row],color="#38484E",lw=.95,ls=":")
            axs[row].plot(f.t[black],angle(f.z["vision_hold_b"])[row][black],color="#98613c",lw=1.05,ls="-.")
    # Shade only when every paired run has exactly the same registered blackout.
    intervals=[contiguous(f.t,f.z["blackout"]) for f in flights.values()]
    shared=bool(intervals) and all(len(v)==len(intervals[0]) and np.allclose(v,intervals[0],atol=.03) for v in intervals[1:])
    for a in axs:
        a.set_xlim(24,max_t);a.grid(axis="y",color=PALE,lw=.45)
        if shared:
            for lo,hi in intervals[0]:a.axvspan(lo,hi,color="#E8ECEF",zorder=-4)
    # The plate focuses on the task interval and the near-frame / alignment
    # ranges. The supplementary matrix retains all times and full error ranges.
    for j in (0,1):
        segments=[angle(f.z[key])[j][f.t>=24] for f in flights.values() for key in ("true_b","est_b","des_b")]
        vals=np.concatenate(segments);vals=vals[np.isfinite(vals)]
        if len(vals):
            margin=max(2.,.06*(vals.max()-vals.min()))
            axs[j].set_ylim(vals.min()-margin,vals.max()+margin)
    axs[2].set_ylim(0,32)
    axs[3].set_ylim(0,16);twin.set_ylim(0,16)
    axs[2].axhline(6.,color=RED,lw=.85,ls="--")
    axs[2].text(.993,6.65,"6.0 cm",va="bottom",ha="right",color=RED,fontsize=7,transform=axs[2].get_yaxis_transform())
    axs[3].axhline(3.5,color="#65797F",lw=.8,ls=":")
    twin.axhline(3.,color=RED,lw=.8,ls="--")
    for a,label in zip(axs,(r"$\theta$ (deg)",r"$\phi$ (deg)","Clearance (cm)","Position error (cm)")):a.set_ylabel(label)
    twin.set_ylabel("Axis error (deg)",color=RED);twin.tick_params(axis="y",colors=RED,labelsize=7.5)
    axs[3].set_xlabel("Time after release (s)")
    handles=[Line2D([],[],color=COLORS[m],lw=1.4,label=NAMES[m]) for m in METHODS]
    fig.legend(handles=handles,ncol=5,loc="upper center",bbox_to_anchor=(.51,.995),frameon=False,fontsize=8)
    fig.text(.095,.918,"θ/φ: physical —, estimated --, fusion reference ···, last visual pose on the fusion flight –·–.",fontsize=7.55,color=MUTED)
    fig.text(.095,.025,"Clearance: arm/tool —, body ···.  After passage: position — (cm), axis -- (deg).  Gray band: 4 s optical occlusion.",fontsize=7.3,color=MUTED)
    save(fig,out,"gate_four_metrics");plt.close(fig)


def keyframes(flights,renderer,out,epochs):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(3,5,figsize=(9.15,4.16),squeeze=False)
    fig.subplots_adjust(left=.035,right=.986,top=.86,bottom=.09,wspace=.035,hspace=.26)
    for j,m in enumerate(METHODS):
        ax[0,j].set_title(NAMES[m],color=COLORS[m],fontsize=9.4,pad=3)
        f=flights.get(m)
        stamps=[epochs[0],None,epochs[-1]]
        if f is not None:
            passage=(f.t>=27.6)&~f.z["gate_exit"].astype(bool)
            ids=np.flatnonzero(passage)
            if len(ids):stamps[1]=float(f.t[ids[np.argmin(f.z["arm_tool_clearance"][ids])]])
        for row,stamp in enumerate(stamps):
            a=ax[row,j];a.set_xticks([]);a.set_yticks([])
            for s in a.spines.values():s.set_visible(False)
            idx=None if f is None or stamp is None else f.index(stamp)
            if idx is None:missing(a,"Outside recorded interval");continue
            goal,axis=renderer.targets[m]
            if row<2:
                points=np.stack([f.z["arm_near_robot"][idx],f.z["arm_near_gate"][idx]])
            else:
                tip=f.z["tip"][idx];tool_axis=f.z["tool_axis"][idx]
                points=np.stack([goal,goal+.18*axis,tip,tip+.18*tool_axis])
            # All five controllers share the same projection and crop in each row.
            # The last row resolves the actual tool, its tip, and both axes at scale.
            region=(([.74,.015,1.99],1.19),([1.0,.01,1.82],.60),([1.87,.205,1.76],.62))[row]
            im,uv=renderer.image(m,stamp,"oblique",points,region)
            if im is None:missing(a,"No physical state at this time");continue
            a.imshow(im)
            a.set_xlim(-.5,im.width-.5);a.set_ylim(im.height-.5,-.5)
            if row<2:
                if uv is not None:
                    a.plot(uv[:,0],uv[:,1],color=RED,lw=1.2);a.scatter(uv[:,0],uv[:,1],s=9,color=RED)
                a.set_xlabel(f"{stamp:.1f} s    |    {100*f.z['arm_tool_clearance'][idx]:.1f} cm",fontsize=7.7,labelpad=1)
            else:
                if uv is not None:
                    a.plot(uv[:2,0],uv[:2,1],color="#A06A36",ls="--",lw=1.25)
                    a.scatter(*uv[0],marker="+",s=34,color="#A06A36")
                    a.plot(uv[2:,0],uv[2:,1],color="#2F7779",lw=1.2)
                    a.scatter(*uv[2],marker="o",s=10,color="#2F7779")
                    if not (0<=uv[2,0]<im.width and 0<=uv[2,1]<im.height):
                        a.text(.5,.91,"Tool outside target view",ha="center",va="top",transform=a.transAxes,fontsize=6.8,color=RED)
                a.set_xlabel(f"{100*f.z['tip_error'][idx]:.1f} cm   |   {f.z['axis_error_deg'][idx]:.1f}°",fontsize=7.7,labelpad=1)
    fig.text(.035,.965,"Gate passage and oblique tool alignment",fontsize=11,weight="bold")
    fig.text(.035,.918,f"Top: dynamic occlusion at {epochs[0]:g} s. Middle: closest arm–frame approach in each flight. Bottom: target alignment at {epochs[-1]:g} s.",fontsize=8.2,color=MUTED)
    fig.text(.035,.013,"Rust segments connect the nearest surfaces. Teal and ochre indicate the measured tool axis and prescribed target direction.",fontsize=7.7,color=MUTED)
    save(fig,out,"gate_keyframes");plt.close(fig)


def video(flights,renderer,out,start,end,fps=8.):
    from PIL import Image,ImageDraw,ImageFont
    from matplotlib.font_manager import findfont
    font=lambda s:ImageFont.truetype(findfont("DejaVu Sans"),s)
    small,normal,title=font(12),font(14),font(18)
    width,height,colw=1500,950,292
    wrist={m:WristFrames(f.folder) for m,f in flights.items()}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    encoder=os.environ.get("GATE_FFMPEG") or shutil.which("ffmpeg")
    if not encoder:
        from imageio_ffmpeg import get_ffmpeg_exe
        encoder=get_ffmpeg_exe()
    p=subprocess.Popen([encoder,"-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{width}x{height}","-r",str(fps),"-i","-","-an","-c:v","libx264","-preset","medium","-crf","21","-pix_fmt","yuv420p","-movflags","+faststart",str(out/"gate_multiview_comparison.mp4")],stdin=subprocess.PIPE)
    def bar(draw,x,y,value,limit,maximum,label,unit):
        draw.text((x,y),f"{label}: {value:.2f} {unit}" if math.isfinite(value) else f"{label}: —",font=small,fill=INK)
        draw.rectangle((x,y+19,x+265,y+24),fill=PALE)
        if math.isfinite(value):
            bad=value<limit if "clearance" in label.lower() else value>limit
            draw.rectangle((x,y+19,x+265*np.clip(value/maximum,0,1),y+24),fill=RED if bad else "#779F98")
        lx=x+265*limit/maximum;draw.line((lx,y+15,lx,y+28),fill=RED,width=2)
    try:
        for now in np.arange(start,end+1e-8,1/fps):
            im=Image.new("RGB",(width,height),"white");d=ImageDraw.Draw(im)
            d.text((17,12),f"Gate feedback comparison   |   time after release {now:.2f} s",font=title,fill=INK)
            d.text((17,41),"Side and oblique CAD views / delivered wrist RGB.   Rust: nearest surfaces.   Teal: measured tool axis.   Ochre: target axis.",font=normal,fill=MUTED)
            for j,m in enumerate(METHODS):
                x=16+j*298;f=flights.get(m);d.text((x,77),NAMES[m],font=title,fill=COLORS[m])
                idx=None if f is None else f.index(now)
                if idx is None:
                    d.text((x,115),"No recorded state at this time",font=small,fill=MUTED);continue
                for view,y in (("side",111),("oblique",301)):
                    dock=now>=40
                    if dock:
                        goal,target_axis=renderer.targets[m];tip=f.z["tip"][idx]
                        points=np.stack((goal,goal+.16*target_axis,tip,tip+.16*f.z["tool_axis"][idx]))
                    else:
                        points=np.stack((f.z["arm_near_robot"][idx],f.z["arm_near_gate"][idx]))
                    # The camera stays fixed across methods at each time. The
                    # second row uses a common instrument view after 40 s.
                    region=(([1.86,.205,1.77],.72) if dock else ([.83,.01,1.96],1.04)) if view=="oblique" else (renderer.cameras["side"]["lookat"],1.52)
                    frame,uv=renderer.image(m,now,view,points,region)
                    if frame is not None:
                        if uv is not None and (dock or f.z["arm_tool_clearance"][idx]<.48):
                            ink=ImageDraw.Draw(frame)
                            pairs=((0,1,"#b5793f"),(2,3,"#268778")) if dock else ((0,1,RED),)
                            for left,right,color in pairs:
                                ends=[tuple(np.rint(uv[k]).astype(int)) for k in (left,right)]
                                ink.line(ends,fill=color,width=3)
                                for px,py in ends[:1] if dock else ends:
                                    ink.ellipse((px-3,py-3,px+3,py+3),fill=color)
                        frame.thumbnail((colw,180));im.paste(frame,(x+(colw-frame.width)//2,y))
                row=wrist[m].at(now)
                if row is not None:
                    frame=wrist[m].image(row);frame.thumbnail((colw,165));im.paste(frame,(x+(colw-frame.width)//2,501))
                    d.text((x,674),f"Wrist exposure {row['t']:.2f} / arrival {row['arrival_t']:.2f} s",font=small,fill=MUTED)
                else:d.text((x,560),"No released visual frame",font=small,fill=MUTED)
                valid=np.isfinite(f.z["est_b"][idx]).all()
                state=str(f.z["phase"][idx]);d.text((x,700),f"State: {state}",font=small,fill=INK)
                badge="NO ESTIMATOR OUTPUT" if not valid else ("PROTECTIVE HOLD" if f.z["hold"][idx] else "Estimator output present")
                d.text((x,721),badge,font=small,fill=RED if not valid or f.z["hold"][idx] else COLORS[m])
                if "dwell_s" in f.z:d.text((x,736),f"Recorded alignment dwell: {f.z['dwell_s'][idx]:.2f} / 1.00 s",font=small,fill=MUTED)
                bar(d,x,755,100*f.z["body_clearance"][idx],6.,18.,"Body clearance","cm")
                bar(d,x,795,100*f.z["arm_tool_clearance"][idx],6.,18.,"Arm/tool clearance","cm")
                bar(d,x,835,100*f.z["tip_error"][idx],3.5,15.,"Tool position error","cm")
                bar(d,x,875,float(f.z["axis_error_deg"][idx]),3.,15.,"Tool axis error","deg")
            d.text((17,927),"Safety ≥ 6.0 cm. Alignment ≤ 3.5 cm and ≤ 3.0 deg for 1.0 continuous second. Black wrist exposures 28–32 s, delivered 0.20 s later.",font=small,fill=MUTED)
            p.stdin.write(im.tobytes())
    except Exception:p.kill();raise
    finally:p.stdin.close();code=p.wait()
    if code:raise RuntimeError(f"ffmpeg failed ({code})")


def overview_asset(out):
    """One static, offline-usable composition, generated only from this invocation."""
    from PIL import Image
    out=Path(out);parts=[]
    for name in ("gate_keyframes.png","gate_four_metrics.png"):
        with Image.open(out/name) as raw:
            image=raw.convert("RGB");image=image.resize((1800,round(image.height*1800/image.width)))
            parts.append(image)
    canvas=Image.new("RGB",(1800,sum(im.height for im in parts)+24),"white")
    y=0
    for im in parts:canvas.paste(im,(0,y));y+=im.height+24
    canvas.save(out/"gate_overview.png")


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root",type=Path,required=True);p.add_argument("--protocol",type=Path)
    p.add_argument("--split",choices=("development","formal"),default="formal")
    p.add_argument("--seed",required=True);p.add_argument("--material",choices=("nominal","differential"),default="differential")
    p.add_argument("--vision",choices=("natural","occluded"),default="occluded")
    p.add_argument("--out",type=Path);p.add_argument("--curves",action="store_true");p.add_argument("--barcode",action="store_true")
    p.add_argument("--keyframes",action="store_true");p.add_argument("--video",action="store_true")
    p.add_argument("--cameras",type=Path);p.add_argument("--model-module",default="gate_feedback_model")
    p.add_argument("--epochs",type=float,nargs="+",help="Predeclared release times shared by all methods")
    p.add_argument("--start",type=float,default=0.);p.add_argument("--end",type=float);p.add_argument("--fps",type=float,default=8.)
    p.add_argument("--allow-incomplete",action="store_true")
    a=p.parse_args()
    if not any((a.curves,a.barcode,a.keyframes,a.video)):p.error("Select one or more explicit output actions")
    proto=a.protocol or a.root/"protocol.json";rows=protocol_cases(proto)
    flights=select_group(a.root,a.split,rows,a.seed,a.material,a.vision,a.allow_incomplete)
    out=a.out or a.root/"review";out.mkdir(parents=True,exist_ok=True);style();stems=[]
    if a.curves:
        curves(flights,out);detailed_curves(flights,out)
        stems.extend(("gate_four_metrics","gate_method_details"))
    if a.barcode:phase_barcode(flights,out);stems.append("gate_state_barcode")
    if a.keyframes or a.video:
        if not a.cameras:p.error("Native rendering requires fixed cameras for the actual new scene")
        if a.keyframes and not a.epochs:p.error("Keyframes require predeclared --epochs; do not select from favorable outcomes")
        renderer=NativeViews(flights,readj(a.cameras),a.model_module)
        try:
            if a.keyframes:keyframes(flights,renderer,out,a.epochs);stems.append("gate_keyframes")
            if a.video:
                end=a.end if a.end is not None else max(f.t[-1] for f in flights.values() if len(f.t))
                if not (a.start<end and a.fps>0):raise ValueError("Invalid video timing")
                video(flights,renderer,out,a.start,end,a.fps)
                stems.append("gate_multiview_comparison")
        finally:renderer.close()
    if a.keyframes and a.curves:overview_asset(out);stems.append("gate_overview")
    inputs=[proto]+[f.folder/"score-trace.npz" for f in flights.values()]
    if a.cameras:inputs.append(a.cameras)
    if a.keyframes or a.video:
        for f in flights.values():
            inputs.extend([p for p in (f.folder/"physics.npz",f.folder/"run.json",f.folder/"optics/index.jsonl",f.folder/"optics/frames.txt") if p.is_file()])
    manifest={"split":a.split,"seed":a.seed,"material":a.material,"vision":a.vision,
        "cases":{m:f.case for m,f in flights.items()},"epochs":a.epochs,"clock":"release seconds; each method has an independent flight",
        "thresholds":{"external_clearance_m":CLEARANCE_M,"tip_position_m":TIP_M,"tool_axis_deg":AXIS_DEG,"continuous_dwell_s":DWELL_S},
        "inputs":[{"path":str(x.resolve()),"sha256":digest(x)} for x in inputs],
        "outputs":[{"path":str(x.resolve()),"sha256":digest(x)} for x in out.glob("gate_*") if x.is_file() and x.stem in stems],
        "model_module":a.model_module,"outcome_source":"independent scorer; no success claims created by visualization"}
    (out/"visual_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":main()
