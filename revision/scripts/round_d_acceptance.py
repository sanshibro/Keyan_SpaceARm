"""D: geometry-only qualification exercising production planner and receiving-map gates.
Independent reference: full C2 MuJoCo collision engine + separately authored physical gate.
"""

import argparse, csv, json, math, sys, time, subprocess, hashlib, platform
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np, mujoco

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "mujoco_closed_loop"))
from planner_adapter import V21Adapter, WholeBodySeedPlanner
from super_metric_model import build_super_metric
from model import SimArmBridge
from super_planning_geometry import PLATFORM, solids
from super_handoff import coordinates, QuinticBendingBridge
from uav_continuum_whole_body_v2.mapping import MapSnapshot
from uav_continuum_whole_body_v2.trajectory import SplineTrajectory
from uav_continuum_whole_body_v2.execution import quintic_bridge_coefficients
from trajectory_tools import reconstruct_spline
from mission_navigation import MissionNavigator


def dump(p, x):
    def json_value(a):
        if isinstance(a, np.ndarray):
            return json_value(a.tolist())
        if isinstance(a, np.generic):
            return json_value(a.item())
        if isinstance(a, dict):
            return {k: json_value(v) for k, v in a.items()}
        if isinstance(a, (list, tuple)):
            return [json_value(v) for v in a]
        if isinstance(a, float) and not math.isfinite(a):
            return "nan" if math.isnan(a) else "+infinity" if a > 0 else "-infinity"
        return a

    p.write_text(json.dumps(json_value(x), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def state(x=0.0, y=0.0, z=1.50, yaw=0.0, theta=0.0, phi=0.0):
    return np.r_[x, y, z, np.deg2rad([yaw, theta, phi])]


def spline(a, b, duration=8.0):
    end = b.copy()
    for k in (3, 5):
        end[k] = a[k] + math.atan2(math.sin(end[k] - a[k]), math.cos(end[k] - a[k]))
    return SplineTrajectory(
        np.linspace(a, end, 4), [duration], stationary_start=True, stationary_end=True
    )


def static(s):
    return spline(s, s, 1.0)


def fixture():
    # Voxel edges, not rendered walls, define the planner input.
    obs = np.indices((64, 60, 55)).reshape(3, -1).T + [-32, -30, 0]
    i, j, k = obs.T
    occupied = (i >= -1) & (i < 1) & ((k < 9) | (k >= 45) | (j < -14) | (j >= 14))
    return MapSnapshot(
        0.05,
        obs[occupied],
        obs,
        frame_id="world",
        source="D_known_voxel_fixture",
        metadata={"platform_id": PLATFORM, "gate": "D1", "test_only": True},
    )


def independent_boxes():
    # Independent analytic specification: x thickness 0.10, y opening 1.40, z [.45,2.25].
    return [
        ("sill", [-0.05, -1.5, 0], [0.05, 1.5, 0.45]),
        ("lintel", [-0.05, -1.5, 2.25], [0.05, 1.5, 2.75]),
        ("jamb_negative", [-0.05, -1.5, 0.45], [0.05, -0.7, 2.25]),
        ("jamb_positive", [-0.05, 0.7, 0.45], [0.05, 1.5, 2.25]),
    ]


class Independent:
    def __init__(self):
        xml, _ = build_super_metric(fixed=False)
        r = ET.fromstring(xml)
        wb = r.find("worldbody")
        wb.remove(wb.find("geom[@name='floor']"))
        # Give the native contacts the same CAD surfaces for external-only checks;
        # their bit mask cannot add robot-robot self contacts.
        for g in wb.find("body[@name='uav']").iter("geom"):
            if (
                g.get("name", "").startswith(("visual_", "qingyu_rviz_"))
                and g.get("contype", "1") == "0"
            ):
                g.set("contype", "4")
                g.set("conaffinity", "0")
        for name, lo, hi in independent_boxes():
            l, h = np.asarray(lo), np.asarray(hi)
            ET.SubElement(
                wb,
                "geom",
                type="box",
                name=name,
                pos=" ".join(map(str, (l + h) / 2)),
                size=" ".join(map(str, (h - l) / 2)),
                contype="2",
                conaffinity="5",
            )
        self.m = mujoco.MjModel.from_xml_string(ET.tostring(r, encoding="unicode"))
        self.d = mujoco.MjData(self.m)
        self.bridge = SimArmBridge()
        self.gate = {self.m.geom(n).id for n, _, _ in independent_boxes()}

    def at(self, s, roll=0.0, pitch=0.0):
        d, m = self.d, self.m
        quat = np.zeros(4)
        # Compose extrinsic yaw, pitch and roll independently of the planning oracle.
        za = np.array([math.cos(s[3] / 2), 0, 0, math.sin(s[3] / 2)])
        ya = np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0])
        xa = np.array([math.cos(roll / 2), math.sin(roll / 2), 0, 0])
        qy = np.zeros(4)
        mujoco.mju_mulQuat(qy, za, ya)
        mujoco.mju_mulQuat(quat, qy, xa)
        d.qpos[:7] = [*s[:3], *quat]
        self.bridge.set_kinematic_shape(m, d, float(np.clip(s[4], 0, math.radians(120))), s[5])
        mujoco.mj_fwdPosition(m, d)
        return [
            (
                float(c.dist),
                m.geom(c.geom1).name,
                m.geom(c.geom2).name,
                "wall" if c.geom1 in self.gate or c.geom2 in self.gate else "self",
            )
            for c in d.contact[: d.ncon]
            if c.dist < -1e-6
        ]

    def sweep(self, sp):
        t = np.unique(np.r_[np.arange(0.0, sp.duration, 0.01), sp.span_boundaries, sp.duration])
        q = sp.evaluate(t)
        hit = None
        n = 0
        for stamp, s in zip(t, q):
            n += 1
            cs = self.at(s)
            if cs:
                hit = dict(time_s=float(stamp), worst=min(cs))
                break
        return dict(penetrated=bool(hit), first=hit, samples=n, dt_s=0.01)


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    mp = fixture()
    mp.save(out / "known_map.npz")
    adapter = V21Adapter(mp)
    ref = Independent()
    orig = json.loads((R / "revision/data/super_geometry_audit.json").read_text())
    cfg = dict(
        platform_id=PLATFORM,
        map_hash=mp.map_hash,
        resolution_m=mp.resolution,
        gate_boxes=independent_boxes(),
        compiled_map_boxes=[dict(name=n, center=c, half=h) for n, c, h in adapter.fine.boxes],
        observed=len(mp.observed_indices),
        occupied=len(mp.occupied_indices),
        yaw_only_tilt_bound_deg=np.rad2deg(adapter.limits.tilt_max),
        body_margin_m=adapter.limits.body_margin,
        arm_margin_m=adapter.limits.arm_margin,
        self_margin_m=0.001,
        model_height_m=adapter.limits.body_height,
        body_center_z_m=adapter.limits.body_center_z,
        external_geom_count=len(adapter.fine.ids),
        compiled_contact_model="C2 CAviz + registered physical proxies",
        candidate_cap_deg=118,
        hard_integrated_cap_deg=120,
        flight=False,
        repo_commit=subprocess.check_output(
            ["git", "-C", str(R), "rev-parse", "HEAD"], text=True
        ).strip(),
        mujoco=mujoco.__version__,
        python=platform.python_version(),
        seed=0,
        input_hashes={
            s: hashlib.sha256((R / s).read_bytes()).hexdigest()
            for s in [
                "revision/data/super_geometry_audit.json",
                "revision/data/super_metric_geometry.json",
                "revision/data/custom_landing_geometry.json",
                "revision/data/body_camera_registration.json",
            ]
        },
    )
    cfg["map_hash"] = str(mp.map_hash)
    cfg.update(
        units={"position": "m", "angles": "rad unless named deg", "time": "s"},
        arm_arc_length_m=adapter.arm.arc_length,
        historical_paper_arm_m=0.22,
        geometry_reference=orig["dimensions"],
        state_order=["world_x", "world_y", "world_z", "body_yaw", "arm_theta", "arm_phi"],
        tracking_reserve_for_adapter_m=0.0,
        extra_reserve_at_mission_receiver_m=0.03,
        self_contact_rule="C2 registered contact pairs; strict 1 mm approach threshold",
        external_clearance_rule="lower bound from AABBs and support-plane projection; known-map exterior and unknown cells blocked",
        validation_steps={
            "candidate_s": 0.04,
            "nominal_translation_m": 0.01,
            "nominal_angle_deg": 0.5,
            "native_s": 0.01,
        },
        pose_reference="prescribed forward kinematics of the C2 candidate assembly; no flight integration",
    )
    dump(out / "config.json", cfg)

    def say(*x):
        print(*x, flush=True)

    grid = []
    a = state(x=-0.8)
    for row in orig["pose_samples"]:
        q = state(x=-0.8, theta=row["theta_deg"], phi=row["phi_deg"])
        contacts = [v for v in ref.at(q) if v[-1] == "self"]
        p = adapter.fine.state(q, True)
        grid.append(
            [
                row["theta_deg"],
                row["phi_deg"],
                int(not row["self_collision_free"]),
                int(bool(contacts)),
                int(p["reason"] == "self_collision"),
            ]
        )
    with (out / "grid.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "theta_deg",
                "phi_deg",
                "historical_penetration",
                "native_penetration",
                "new_1mm_rejection",
            ]
        )
        w.writerows(grid)
    say(
        "GRID: saved/independent penetrations, 1 mm rejects",
        np.array(grid)[:, 2:].sum(axis=0).tolist(),
    )
    checks = []
    failures = []

    def add(label, want, got, meta=None):
        item = {
            "case": label,
            "expected": bool(want),
            "accepted": bool(got),
            "match": bool(want) == bool(got),
            "detail": meta or {},
        }
        checks.append(item)
        if not item["match"]:
            failures.append(label)
        say("CASE", label, item["accepted"], "expected", item["expected"])
        dump(out / "cases.json", checks)

    add(
        "48 archived penetration labels",
        True,
        all(a[2] == a[3] for a in grid),
        {
            "points": len(grid),
            "hard_hits": sum(p[3] for p in grid),
            "one_mm_rejects": sum(p[4] for p in grid),
        },
    )
    # Independently count and raster-check the represented union (not just its volume).
    b, lo, hi = solids(mp)
    cent = mp.observed_points
    inside = np.zeros(len(cent), bool)
    for _, ctr, hs in b:
        inside |= (np.abs(cent - ctr) < hs + 1e-10).all(axis=1)
    add(
        "voxel union matches all cell centres",
        True,
        np.array_equal(mp.observed_indices[inside], mp.occupied_indices),
    )
    tiny = np.indices((3, 3, 3)).reshape(3, -1).T
    tiny = tiny[np.any(tiny != [1, 1, 1], axis=1)]
    unknown, _, _ = solids(MapSnapshot(0.05, np.empty((0, 3), int), tiny))
    add(
        "unobserved interior cell becomes solid",
        True,
        len(unknown) == 1
        and unknown[0][0].startswith("unknown")
        and np.allclose(unknown[0][1], [0.075] * 3)
        and np.allclose(unknown[0][2], [0.025] * 3),
    )
    begin = state(x=-0.8, theta=0, phi=90)
    end = state(x=-0.8, theta=90, phi=90)
    pos = state(x=-0.8, theta=0, phi=180)
    bent = state(x=-0.8, theta=118, phi=180)
    safe = spline(pos, bent)
    unsafe = spline(begin, end)
    wide_a = state(x=-0.48, theta=0, phi=180)
    wide_b = state(x=0.48, theta=0, phi=180)
    gatepath = spline(wide_a, wide_b)
    # The negative paths are specified candidates, not claims that the whole gate is infeasible.
    skid_a = state(x=-0.48, z=0.80, theta=118, phi=180)
    skid_b = state(x=0.48, z=0.80, theta=118, phi=180)
    tool_a = state(x=-0.48, z=0.88)
    tool_b = state(x=0.48, z=0.88)
    # Aligned separated skid and sill: the two x faces are 279.99 mm apart.
    geom = adapter.fine
    idx = int(np.flatnonzero(geom.ids == geom.m.geom("collision_landing_151").id)[0])
    geom.place(geom.w, skid_a)
    mujoco.mj_kinematics(geom.m, geom.w)
    lower = 0.48 - 0.05 - 0.15001014
    measure = geom.component_map_clearance(idx, 1, lower, 0.5)
    add(
        "aligned skid to sill support planes",
        True,
        abs(measure - lower) < 1e-5,
        {"analytic_m": lower, "computed_m": measure},
    )
    margin_only = state(x=0.0, z=1.04)
    protected = geom.state(margin_only, True)
    unbuffered = ref.at(margin_only)
    add(
        "positive separation can fail the reserved margin",
        True,
        protected["reason"] in ("arm_environment", "body_environment") and not unbuffered,
        {
            "reason": protected["reason"],
            "raw_arm_m": protected["raw_arm"],
            "native_contacts": len(unbuffered),
        },
    )
    report = adapter.validator.validate(unsafe)
    add(
        "production validator rejects interior collision",
        False,
        report.feasible,
        {"reason": report.failure_reason},
    )
    impossible = state(x=-0.72, theta=60, phi=90)
    rejected, details = adapter.plan(
        begin, impossible[:3], keep_yaw=True, terminal_shape=impossible[4:]
    )
    add(
        "fully constrained nearby terminal rejected",
        True,
        rejected is None and details["status"] == "terminal_geometry_rejected",
        {"status": details["status"]},
    )
    false_accepts = []
    native_accepted = []
    paths = [
        ("self sweep phi90", unsafe, False),
        ("front bend", safe, True),
        ("front return", spline(bent, pos), True),
        (
            "periodic 179 to -179",
            spline(state(x=-0.8, theta=90, phi=179), state(x=-0.8, theta=90, phi=-179)),
            True,
        ),
        ("wide gate level route", gatepath, True),
        ("low route skid contact", spline(skid_a, skid_b), False),
        ("low route tool contact", spline(tool_a, tool_b), False),
    ]
    for name, sp, want in paths:
        t = time.monotonic()
        proof = adapter.fine.proof(sp)
        wall = time.monotonic() - t
        oracle = ref.sweep(sp)
        if proof["accepted"]:
            native_accepted.append(name)
            if oracle["penetrated"]:
                false_accepts.append(name)
        add(name, want, proof["accepted"], dict(proof=proof, native=oracle, wall_s=wall))
    add(
        "all frozen negative routes contain a physical witness",
        True,
        all(
            next(x for x in checks if x["case"] == n)["detail"]["native"]["penetrated"]
            for n in ("self sweep phi90", "low route skid contact", "low route tool contact")
        ),
    )
    for name, st in [("skid", state(z=0.80, theta=118, phi=180)), ("tool", state(z=0.88))]:
        contacts = ref.at(st)
        specific = [
            v
            for v in contacts
            if v[-1] == "wall"
            and any(
                w in (" ".join(v[1:3]))
                for w in (
                    ("landing_",) if name == "skid" else ("gripper", "d435", "finger", "wrist")
                )
            )
        ]
        add(
            "named " + name + " at gate centre",
            True,
            bool(specific),
            {"witness": min(specific) if specific else None},
        )
    seed = WholeBodySeedPlanner(
        adapter.world,
        adapter.arm,
        adapter.mass,
        adapter.limits,
        adapter.config,
        platform_geometry=adapter.proposal,
    )
    add("seed rejects interior hit", False, np.isfinite(seed._edge_cost(begin, end)))
    add("seed keeps front bend", True, np.isfinite(seed._edge_cost(pos, bent)))
    for name, st, ed, want in [
        ("short unsafe", begin, end, False),
        ("short clear", pos, state(x=-0.70, theta=0, phi=180), True),
    ]:
        tr, _ = adapter.short_direct_bridge(st, ed, time.monotonic(), 15.0)
        add(name, want, tr is not None, {"last_rejection": adapter.last_short_rejection})
    z = np.zeros(6)

    def qbridge(st, ed):
        return QuinticBendingBridge(
            quintic_bridge_coefficients(coordinates(st), z, z, coordinates(ed), z, z, 8.0), 8.0
        )

    for name, st, ed, bridge, want in [
        ("direct qualified", wide_a, wide_a, None, True),
        ("direct boundary mismatch", begin, end, None, False),
        ("bridge crosses skid", begin, end, qbridge(begin, end), False),
        ("bridge clear", pos, bent, qbridge(pos, bent), True),
    ]:
        candidate = static(ed)
        p = adapter.handoff_review(st, z, candidate, bridge, start_acceleration=z)
        add(
            name,
            want,
            p.feasible,
            dict(
                reason=p.failure_reason,
                record=p.metrics.get("transition", p.metrics.get("handoff_geometry", {})),
            ),
        )
        if p.feasible:
            native_accepted.append(name)
            if ref.sweep(bridge or candidate)["penetrated"] or ref.sweep(candidate)["penetrated"]:
                false_accepts.append(name)
    bad = state(x=-0.8, theta=119)
    straight = state(x=-0.8, theta=0, phi=0)
    neutral_switch = adapter.handoff_review(straight, z, static(state(x=-0.8, theta=0, phi=179)))
    add("straight-arm plane is unobservable at handoff", True, neutral_switch.feasible)
    timed_out, timeout_info = adapter.plan(wide_a, wide_b[:3], budget=0.0, keep_yaw=True)
    add(
        "expired planning budget holds the buffer",
        True,
        timed_out is None and not timeout_info.get("success", False),
        {"status": timeout_info["status"], "reason": timeout_info.get("error")},
    )
    wrong_acc = np.array([0.1, 0, 0, 0, 0, 0])
    wrong_bridge = QuinticBendingBridge(
        quintic_bridge_coefficients(coordinates(pos), z, wrong_acc, coordinates(pos), z, z, 8.0),
        8.0,
    )
    bad_boundary = adapter.handoff_review(pos, z, static(pos), wrong_bridge, start_acceleration=z)
    add(
        "quintic start acceleration mismatch",
        True,
        not bad_boundary.feasible and bad_boundary.failure_reason == "bridge_acceleration_mismatch",
    )
    tr, log = adapter.plan(pos, [-0.68, 0, 1.5], terminal_shape=bad[4:6])
    add("119 degree command never admitted", False, tr is not None, {"status": log["status"]})
    # A separate fine angular pass checks the first witness and refinement stability.
    refine = []
    for step in [1.0, 0.5, 0.25]:
        t = time.monotonic()
        p = adapter.fine.proof(
            unsafe, dt=0.12 if step == 1.0 else 0.06 if step == 0.5 else 0.03, da=math.radians(step)
        )
        w = p["first"]
        refine.append(
            dict(
                step_deg=step,
                samples=p["samples"],
                first_time_s=w["time_s"],
                first_theta_deg=math.degrees(w["state"][4]),
                reason=w["reason"],
                pair=w["pair"],
                wall_s=time.monotonic() - t,
            )
        )
    add(
        "finer grids retain interior rejection",
        True,
        all(i["reason"] == "self_collision" for i in refine),
        {"refinement": refine},
    )
    # Independent explicit attitude check at the accepted level-route positions.
    tilt_hits = []
    for t, s in zip(
        np.linspace(0, gatepath.duration, 41),
        gatepath.evaluate(np.linspace(0, gatepath.duration, 41)),
    ):
        for roll, pitch in [(15, 0), (-15, 0), (0, 15), (0, -15)]:
            if ref.at(s, math.radians(roll), math.radians(pitch)):
                tilt_hits.append([float(t), roll, pitch])
    add(
        "wide gate +/-15 degree cardinal attitude probes",
        True,
        not tilt_hits,
        {"evaluations": 164, "hits": tilt_hits},
    )
    # Exercise the actual reference/beam-seed/finite-budget optimizer, not only a handcrafted spline.
    logs = []

    def progress(stage, info):
        logs.append(stage)
        say("PLANNER", stage)

    tt = time.monotonic()
    tr, info = adapter.plan(
        wide_a,
        wide_b[:3],
        budget=15.0,
        keep_yaw=True,
        terminal_shape=wide_b[4:6],
        progress=progress,
    )
    elapsed = time.monotonic() - tt
    dump(out / "planner_run.json", info)
    full = None
    if tr is not None:
        sp = reconstruct_spline(tr)
        recv = adapter.handoff_review(wide_a, np.zeros(6), sp)
        native = ref.sweep(sp)
        nav = MissionNavigator(
            out / "known_map.npz", out, "receiver", wide_a, wide_b[:3], goal_shape=wide_b[4:]
        )
        received = {}
        installed = nav.receive_candidate(tr, wide_a, z, received)
        # Inject a spline that would pass an endpoint-only test at the real buffer entry.
        injected = unsafe.sample(
            0.05,
            source="negative_control",
            metadata={"span_durations": [8.0], "conditioned_initial_rate": None},
        )
        refusal = {}
        replaced = nav.receive_candidate(injected, begin, z, refusal)
        add(
            "buffer rejects unsafe replacement and retains accepted candidate",
            True,
            installed and not replaced and nav.trajectory is tr,
            {
                "reason": refusal.get("full_sensor_map_validation", {}).get("failure_reason"),
                "extra_receiver_reserve_m": nav.tracking_reserve,
            },
        )
        full = dict(
            accepted=recv.feasible and installed,
            native=native,
            proof=recv.metrics.get("super_geometry"),
            buffer_proof=received.get("full_sensor_map_validation"),
            elapsed_s=elapsed,
            wall_budget_s=15.0,
            samples=len(tr.times),
        )
        dump(out / "receiver.json", full)
        nav.planner.close()
        if recv.feasible:
            native_accepted.append("end-to-end planner")
            false_accepts += ["end-to-end planner"] if native["penetrated"] else []
        np.savetxt(
            out / "planned_states.csv",
            np.column_stack((tr.times, tr.states)),
            delimiter=",",
            header="candidate_t_s,x_m,y_m,z_m,yaw_rad,theta_rad,phi_rad",
            comments="",
        )
    add(
        "full planner to receiving buffer",
        True,
        full is not None and full["accepted"] and elapsed <= 15.0 and "seed_ready" in logs,
        {
            "status": info["status"],
            "error": info.get("error"),
            "elapsed_s": elapsed,
            "stages": logs,
            "candidate": full,
        },
    )
    add(
        "no native penetration among accepted paths",
        True,
        not false_accepts and len(native_accepted) >= 5,
        dict(paths=native_accepted, false_accepts=false_accepts),
    )
    with (out / "cases.csv").open("w", newline="") as stream:
        fields = [
            "case",
            "expected_condition",
            "observed_condition",
            "matched",
            "failure_reason",
            "first_time_s",
            "first_pair",
            "body_clearance_m",
            "arm_clearance_m",
            "native_penetration",
            "wall_s",
        ]
        table = csv.DictWriter(stream, fieldnames=fields)
        table.writeheader()
        for row in checks:
            detail = row["detail"]
            proof = detail.get("proof") or {}
            first = proof.get("first") or {}
            table.writerow(
                dict(
                    case=row["case"],
                    expected_condition=row["expected"],
                    observed_condition=row["accepted"],
                    matched=row["match"],
                    failure_reason=first.get("reason") or detail.get("reason"),
                    first_time_s=first.get("time_s"),
                    first_pair="/".join(first.get("pair") or []),
                    body_clearance_m=proof.get("body_min"),
                    arm_clearance_m=proof.get("arm_min"),
                    native_penetration=(detail.get("native") or {}).get("penetrated"),
                    wall_s=detail.get("wall_s", detail.get("elapsed_s")),
                )
            )
    summary = dict(
        passed=not failures,
        cases=len(checks),
        failures=failures,
        independent_accepted_paths=len(native_accepted),
        false_accepts=false_accepts,
        archived_grid_penetrations=sum(a[2] for a in grid),
        native_grid_penetrations=sum(a[3] for a in grid),
        new_grid_1mm_rejections=sum(a[4] for a in grid),
        planner_status=info["status"],
        planner_wall_s=elapsed,
        planner_budget_s=15.0,
        geometry_handoff_only=True,
        flight_commands_sent=0,
        wall_total_s=time.monotonic() - start,
        refinement=refine,
        proposal_calls=adapter.proposal.calls,
        final_calls=adapter.fine.calls,
    )
    dump(out / "summary.json", summary)
    say("ROUND_D_COMPLETE", json.dumps(summary))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    main(p.parse_args().out)
