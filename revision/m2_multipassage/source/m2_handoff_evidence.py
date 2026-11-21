"""Time-align wrist exposures, rolling handoff clocks and measured PX4/MuJoCo response."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize_switches(case, recovery_s=.65):
    flight=case/'flight'
    with np.load(flight/'control.npz', allow_pickle=False) as c, np.load(case/'audit/series.npz', allow_pickle=False) as a:
        t=c['t']; acceleration=np.linalg.norm(c['plan_ddq'][:,:3],axis=1)
        if not np.array_equal(t,a['t']):raise ValueError('Review clocks do not match')
        rotations=Rotation.from_quat(a['qpos'][:,[4,5,6,3]]).as_matrix()
        tilt=np.rad2deg(np.arccos(np.clip(rotations[:,2,2],-1,1)))
    result=[]
    for e in read_lines(flight/'rolling/events.jsonl'):
        if e['event']!='activate' or e['source_id']<0:continue
        start=e['bridge_start_s'];end=e['bridge_end_s']
        bridge=(t>=start)&(t<=end)
        response=(t>=start)&(t<=end+recovery_s)
        result.append(dict(window=e['id'],start_s=start,end_s=end,kind=e['handoff']['kind'],
            peak_planned_translation_accel_m_s2=float(acceleration[bridge].max()),
            peak_measured_tilt_including_recovery_deg=float(tilt[response].max())))
    m=read_json(case/'audit/metrics.json')
    return dict(seed=m['seed'],method=m['strategy'],development=m['development'],physical_ulog_sha256=m['raw_ulog_sha256'],
                recovery_interval_s=recovery_s,external_clearance_m=m['external_lower_m'],bridges=result)


def render(case, output, frame_ids):
    flight = case / "flight"
    with np.load(flight / "control.npz", allow_pickle=False) as control:
        t = control["t"]
        plan = control["plan_q"]
        accel = control["plan_ddq"][:, :3]
        desired = control["des_b"]
        ref = control["ref_world"]
        phase = control["window_status"]
        control_valid = control["valid"]
    with np.load(case / "audit" / "series.npz", allow_pickle=False) as series:
        position = series["qpos"][:, :3]
        quaternion = series["qpos"][:, [4, 5, 6, 3]]
        measured_bend = np.rad2deg(series["theta"])
        thrust = series["rotor_N"]
        if not np.array_equal(t, series["t"]):
            raise ValueError("Independent audit and control clocks differ")
    run = read_json(flight / "run.json")
    release = run["release_us"]
    with np.load(flight / "physics.npz", allow_pickle=False) as physics:
        source_time = physics["us"].copy()
        verify = np.searchsorted(source_time, release + np.rint(t * 1e6).astype(np.int64))
        if not np.array_equal(source_time[verify], release + np.rint(t * 1e6).astype(np.int64)):
            raise ValueError("Native timestamps do not coincide")
        if not np.allclose(position, physics["qpos"][verify, :3], rtol=0, atol=1e-12):
            raise ValueError("Replay position and independent score differ")
    attitude = Rotation.from_quat(quaternion)
    tilt = np.rad2deg(np.arccos(np.clip(attitude.as_matrix()[:, 2, 2], -1, 1)))
    yaw = np.unwrap(attitude.as_euler("xyz")[:, 2])
    heading_error = np.rad2deg(np.arctan2(np.sin(plan[:, 3]-yaw), np.cos(plan[:, 3]-yaw)))
    accel_norm = np.linalg.norm(accel, axis=1)
    arm = np.rad2deg(np.linalg.norm(desired, axis=1))
    tracking = 100 * np.linalg.norm(position - ref, axis=1)
    # Independent, direct actuator channels (PX4 order, native plant thrust units).
    span = np.ptp(thrust, axis=1)
    exposures = {row["frame"]: row for row in read_lines(flight / "optics" / "index.jsonl")}
    events = read_lines(flight / "rolling" / "events.jsonl")
    optics = read_lines(flight / "optics" / "optical.jsonl")
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, len(frame_ids), figsize=(8.3, 6.0), sharex="col", squeeze=False)
    colours = {"command": "#ad612d", "response": "#146166", "blue": "#305e80"}
    evidence = []
    def values(i):
        return dict(time_s=float(t[i]), position_m=position[i].tolist(), reference_m=ref[i].tolist(),
                    requested_accel_m_s2=float(accel_norm[i]), actual_tilt_deg=float(tilt[i]),
                    planned_bend_deg=float(arm[i]), actual_bend_deg=float(measured_bend[i]),
                    position_error_cm=float(tracking[i]), yaw_error_deg=float(heading_error[i]),
                    rotor_max_minus_min_N=float(span[i]), rotor_N=thrust[i].tolist())
    for col, frame_id in enumerate(frame_ids):
        e = exposures[frame_id]
        shot = e["t"]
        switch = next(row for row in events if row["event"] == "activate" and
                      row.get("bridge_start_s", float("inf")) <= shot <= row.get("bridge_end_s", -1))
        start, end = switch["bridge_start_s"], switch["bridge_end_s"]
        before = switch["handoff"]["before"]
        after = switch["handoff"]["after"]
        window = (t >= start - .60) & (t <= end + 1.0)
        bridge = (t >= start) & (t <= end)
        tail = (t >= start) & (t <= end + .65)
        steady = np.flatnonzero(phase == "stopped")
        stops = []
        for group in np.split(steady, np.flatnonzero(np.diff(steady) > 1) + 1):
            if len(group) and start - 5 <= t[group[-1]] <= start:
                stops.append([float(t[group[0]]), float(t[group[-1]] + np.median(np.diff(t)))])
        inliers = [o for o in optics if start <= o["arrival_t"] <= end]
        row = dict(frame=frame_id, exposure_s=shot, delivery_s=e["arrival_t"],
                   handoff=dict(source=switch["source_id"], receiving=switch["id"], start_s=start,
                                end_s=end, kind=switch["handoff"]["kind"],
                                endpoint_translation_m=(np.asarray(after["q"][:3])-before["q"][:3]).tolist(),
                                endpoint_yaw_change_deg=float(np.rad2deg(after["q"][3]-before["q"][3])),
                                endpoint_bend_deg=np.rad2deg([before["q"][4],after["q"][4]]).tolist()),
                   preceding_stops_s=stops, screenshot=values(int(np.argmin(abs(t-shot)))),
                   at_start=values(int(np.argmin(abs(t-start)))),
                   at_end=values(int(np.argmin(abs(t-end)))),
                   t_and_max_response={key:[float(t[i]),float(data[i])] for key,data,mask in
                        (("planned_accel_m_s2",accel_norm,bridge),("tilt_deg",tilt,tail),
                         ("rotor_spread_N",span,tail),("tracking_error_cm",tracking,tail))
                        for i in [int(np.where(mask,data,-np.inf).argmax())]},
                   valid_feedback_all=bool(np.all(control_valid[bridge])),
                   arrived_visual_frames=len(inliers), accepted_visual_frames=sum(bool(o["valid"]) for o in inliers),
                   contact_steps_entire_flight=run["contact_integration_steps"])
        evidence.append(row)
        for ax in axes[:, col]:
            ax.axvspan(start, end, color="#dce9ec", zorder=0)
            ax.axvline(shot, color="#37464a", lw=.8, ls=(0,(2,2)))
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y", color="#e4e8e7", lw=.6)
            ax.tick_params(labelsize=8, length=3)
        axes[0,col].plot(t[window], accel_norm[window], color=colours["command"], lw=1.7)
        axes[1,col].plot(t[window], tilt[window], color=colours["response"], lw=1.7)
        axes[2,col].plot(t[window], arm[window], color=colours["command"], lw=1.3, ls="--", label="Command")
        axes[2,col].plot(t[window], measured_bend[window], color=colours["response"], lw=1.6, label="Measured")
        axes[0,col].set_title(f"({'abcdefghijklmnopqrstuvwxyz'[col]})   {start:.2f}–{end:.2f} s handoff",
                              loc="left", fontsize=11, fontweight="bold")
        axes[2,col].set_xlabel("Flight time (s)", fontsize=9)
        axes[2,col].set_xlim(start-.6,end+1.0)
        axes[2,col].set_xticks(np.arange(np.ceil((start-.6)*2)/2, end+1.001, .5))
    for r, label in enumerate(["Commanded translation\nacceleration (m/s²)", "Measured body tilt (°)",
                               "Arm bending angle (°)"]):
        axes[r,0].set_ylabel(label, fontsize=9)
        lower=min(ax.get_ylim()[0] for ax in axes[r]);upper=max(ax.get_ylim()[1] for ax in axes[r])
        for ax in axes[r]:
            ax.set_ylim(0 if r<2 else lower,upper)
    axes[2,-1].legend(loc="best", frameon=False, fontsize=8)
    fig.subplots_adjust(left=.13,right=.965,top=.89,bottom=.14,hspace=.22,wspace=.19)
    fig.text(.13,.961,"Native flight response at the two wrist exposures",fontweight="bold",fontsize=14)
    fig.text(.13,.035,"Shaded interval: quintic handoff.   Dotted line: original wrist exposure.",
             fontsize=8.5,color="#34454a")
    fig.savefig(output / "handoff-response.png",dpi=210,facecolor="white")
    fig.savefig(output / "handoff-response.pdf",facecolor="white")
    plt.close(fig)
    (output/"sources-and-measurements.json").write_text(json.dumps(evidence,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return evidence


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("case",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("frames",type=int,nargs="+")
    parser.add_argument('--comparison-case', type=Path)
    args=parser.parse_args()
    print(json.dumps(render(args.case,args.output,args.frames),ensure_ascii=False))
    if args.comparison_case:
        comparison={k:summarize_switches(s) for k,s in [('original',args.case),('updated',args.comparison_case)]}
        if comparison['original']['seed']!=comparison['updated']['seed']:raise ValueError('Pair requires the same seed')
        (args.output/'paired-development-handoffs.json').write_text(json.dumps(comparison,indent=2)+'\n',encoding='utf-8')
