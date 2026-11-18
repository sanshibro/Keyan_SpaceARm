"""Opt-in frozen voxel-map geometry for the SUPER/Qingyu development platform.
A private FK model is used only on the planning thread; it never writes live physics.
Proposal: yawed component AABBs, conservative. Acceptance: compiled convex geoms
against the union of occupied/unknown voxel boxes, plus native self contacts.
"""

import math, time, xml.etree.ElementTree as ET
from collections import OrderedDict
import numpy as np
import mujoco
from model import SimArmBridge
from super_metric_model import build_super_metric
from shape_limits import COMMAND_THETA_MAX

PLATFORM = "super_qingyu_metric_candidate_v1"
SELF_GAP = 0.001


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_matrix(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def runs(mask, offset, scale, kind, limit):
    """Losslessly tile a binary voxel mask with disjoint cuboids."""
    m = mask.copy()
    boxes = []
    for i, j, k in np.argwhere(m):
        if not m[i, j, k]:
            continue
        k1 = k + 1
        while k1 < m.shape[2] and m[i, j, k1]:
            k1 += 1
        j1 = j + 1
        while j1 < m.shape[1] and m[i, j1, k:k1].all():
            j1 += 1
        i1 = i + 1
        while i1 < m.shape[0] and m[i1, j:j1, k:k1].all():
            i1 += 1
        m[i:i1, j:j1, k:k1] = False
        low = (np.array([i, j, k]) + offset) * scale
        high = (np.array([i1, j1, k1]) + offset) * scale
        boxes.append((kind + "_" + str(len(boxes)), (low + high) / 2, (high - low) / 2))
        if len(boxes) > limit:
            raise ValueError("SUPER frozen map exceeds the cuboid budget; hold")
    return boxes


def solids(snapshot):
    obs = snapshot.observed_indices
    occ = snapshot.occupied_indices
    if not len(obs):
        raise ValueError("SUPER map has no observed cells")
    all_cells = np.vstack((obs, occ))
    lo = np.min(all_cells, axis=0)
    hi = np.max(all_cells, axis=0) + 1
    dims = tuple(map(int, hi - lo))
    if math.prod(dims) > 1500000:
        raise ValueError("SUPER frozen map exceeds the voxel budget; hold")
    known = np.zeros(dims, bool)
    known[tuple((obs - lo).T)] = True
    occupied = np.zeros_like(known)
    occupied[tuple((occ - lo).T)] = True
    boxes = runs(occupied, lo, snapshot.resolution, "occupied", 512)
    boxes += runs(~(known | occupied), lo, snapshot.resolution, "unknown", 512 - len(boxes))
    return boxes, lo * snapshot.resolution, hi * snapshot.resolution


def at_most(theta):
    return -1e-10 <= theta <= COMMAND_THETA_MAX + 1e-10


def shape_span(a, b):
    return max(abs(b[4] - a[4]), abs(wrap(b[5] - a[5])) * min(1.0, max(abs(a[4]), abs(b[4]))))


def sampled_times(spline, dt=0.04, dp=0.01, da=math.radians(0.5)):
    if not np.isfinite(spline.duration) or spline.duration <= 0 or spline.duration > 900:
        raise ValueError("invalid geometry proof duration")
    base = np.unique(
        np.r_[spline.span_boundaries, np.arange(0.0, spline.duration, dt), spline.duration]
    )
    q = spline.evaluate(base, 0)
    if np.shape(q) != (len(base), 6) or not np.isfinite(q).all():
        raise ValueError("non-finite geometry path")
    out = [float(base[0])]
    for i, (a, b) in enumerate(zip(q[:-1], q[1:])):
        n = max(
            1,
            int(
                math.ceil(
                    max(
                        np.linalg.norm(b[:3] - a[:3]) / dp,
                        abs(b[3] - a[3]) / da,
                        shape_span(a, b) / da,
                    )
                )
            ),
        )
        if n > 2000 or len(out) + n > 100000:
            raise ValueError("geometry proof sample budget exceeded")
        out.extend(np.linspace(base[i], base[i + 1], n + 1)[1:].tolist())
    return np.asarray(out)


class SuperGeometry:
    def __init__(self, snapshot, limits):
        self.map_hash = snapshot.map_hash
        self.limits = limits
        self.boxes, self.lo, self.hi = solids(snapshot)
        xml, info = build_super_metric(fixed=False)
        root = ET.fromstring(xml)
        world = root.find("worldbody")
        world.remove(world.find("geom[@name='floor']"))
        for name, pos, hs in self.boxes:
            ET.SubElement(
                world,
                "geom",
                name="Dmap_" + name,
                type="box",
                pos=" ".join(map(str, pos)),
                size=" ".join(map(str, hs)),
                contype="0",
                conaffinity="0",
                rgba="0 0 0 0",
            )
        self.m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        m = self.m
        self.fk = mujoco.MjData(m)
        self.w = mujoco.MjData(m)
        self.bridge = SimArmBridge()
        self.u = m.body("uav").id
        self.root = int(m.jnt_qposadr[int(m.body_jntadr[self.u])])
        self.names = [
            m.geom(i).name
            or ("mount_drive" if m.geom_bodyid[i] == m.body("mount").id else "geom_" + str(i))
            for i in range(m.ngeom)
        ]
        # The outside-world check includes both registered physical proxies and the
        # convex hulls of displayed CAD meshes, so visible surfaces are not lost.
        self.ids = np.array(
            [
                i
                for i in range(m.ngeom)
                if m.geom_bodyid[i] != 0
                and (
                    m.geom_contype[i]
                    or m.geom_conaffinity[i]
                    or (m.geom_rgba[i, 3] > 0.001 and m.geom_group[i] in (1, 2))
                )
            ],
            dtype=int,
        )
        self.groups = np.isin(m.geom_bodyid[self.ids], [self.u, m.body("mount").id])
        self.body = self.ids[self.groups]
        self.tool = self.ids[~self.groups]
        self.eg = np.array([m.geom("Dmap_" + b[0]).id for b in self.boxes], int)
        self.ec = np.array([b[1] for b in self.boxes]).reshape(-1, 3)
        self.eh = np.array([b[2] for b in self.boxes]).reshape(-1, 3)
        self.ab = m.geom_aabb[self.ids].copy()
        self.cache = OrderedDict()
        self.fastcache = OrderedDict()
        m.geom_margin[:] = SELF_GAP
        m.geom_gap[:] = 0
        self.info = info["runtime_geometry"]
        self.radius = info["prop_sweep_body_radius_m"]
        self.theta_max = COMMAND_THETA_MAX
        self.mesh_vertices = {}
        for geom in self.ids:
            if m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = int(m.geom_dataid[geom])
                begin = int(m.mesh_vertadr[mesh])
                self.mesh_vertices[geom] = m.mesh_vert[begin : begin + m.mesh_vertnum[mesh]]
        self.calls = {"search": 0, "final": 0, "seed_edges": 0}
        self.deadline = None

    def check_deadline(self):
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise TimeoutError("SUPER geometry budget exhausted before acceptance")

    def place(self, d, state, RP=(0.0, 0.0)):
        r, p = RP
        cy, sy = math.cos(state[3] / 2), math.sin(state[3] / 2)
        cp, sp = math.cos(p / 2), math.sin(p / 2)
        cr, sr = math.cos(r / 2), math.sin(r / 2)
        d.qpos[self.root : self.root + 7] = [
            *state[:3],
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
        self.bridge.set_kinematic_shape(
            self.m, d, float(np.clip(state[4], 0, self.theta_max)), state[5]
        )

    def local(self, theta, phi):
        key = (float(theta), wrap(phi) if theta > 1.0e-10 else 0.0)
        if key in self.cache:
            return self.cache[key]
        d = self.fk
        self.place(d, [0, 0, 0, 0, *key])
        mujoco.mj_fwdPosition(self.m, d)
        contacts = [
            (float(c.dist), self.names[c.geom1], self.names[c.geom2])
            for c in d.contact[: d.ncon]
            if c.dist < SELF_GAP
        ]
        mat = d.geom_xmat[self.ids].reshape(-1, 3, 3)
        ctr = d.geom_xpos[self.ids] + np.einsum("nij,nj->ni", mat, self.ab[:, :3])
        # Exact box for this component in the local axes of its actual compiled geom.
        radius = np.linalg.norm(ctr, axis=1) + np.linalg.norm(self.ab[:, 3:], axis=1)
        v = (ctr.copy(), mat.copy(), radius, min(contacts) if contacts else None)
        if len(self.cache) >= 512:
            self.cache.popitem(last=False)
        self.cache[key] = v
        return v

    def component_map_clearance(self, index, box, aabb_gap, cutoff):
        """Use a witness direction to certify separation by projecting both solids.

        A clipped or inaccurate native distance alone cannot establish clearance.
        The public MuJoCo build used here can halve a positive box--box distance.
        A Euclidean AABB gap and the gap between support planes are lower bounds.
        """
        geom = int(self.ids[index])
        witness = np.zeros(6)
        native = mujoco.mj_geomDistance(self.m, self.w, geom, int(self.eg[box]), cutoff, witness)
        direction = witness[:3] - witness[3:]
        length = np.linalg.norm(direction)
        certified = max(0.0, aabb_gap)
        if length > 1e-10:
            direction /= length
            rotation = self.w.geom_xmat[geom].reshape(3, 3)
            local = rotation.T @ direction
            size = self.m.geom_size[geom]
            kind = self.m.geom_type[geom]
            if geom in self.mesh_vertices:
                support = self.mesh_vertices[geom] @ local
                left, right = float(support.min()), float(support.max())
            else:
                offset = 0.0
                if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                    radius = size[0]
                elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    radius = size[0] + size[1] * abs(local[2])
                elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    radius = size[0] * np.linalg.norm(local[:2]) + size[1] * abs(local[2])
                elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                    radius = np.linalg.norm(size * local)
                else:
                    offset = self.ab[index, :3] @ local
                    radius = self.ab[index, 3:] @ np.abs(local)
                left, right = offset - radius, offset + radius
            delta = (self.w.geom_xpos[geom] - self.ec[box]) @ direction
            wall_radius = self.eh[box] @ np.abs(direction)
            certified = max(certified, delta + left - wall_radius, -delta - right - wall_radius)
        if certified > 0.0:
            return float(certified)
        return float(min(0.0, native))

    def state(self, state, exact=False):
        self.check_deadline()
        s = np.asarray(state, float)
        if s.shape != (6,) or not np.isfinite(s).all() or not at_most(s[4]):
            return dict(
                body=-1.0,
                arm=-1.0,
                raw_body=-1.0,
                raw_arm=-1.0,
                reason="command_range",
                pair=["command", "theta"],
                self_contact=None,
            )
        k = s.tobytes()
        if not exact and k in self.fastcache:
            return self.fastcache[k]
        self.calls["final" if exact else "search"] += 1
        ctr, mat, rad, hit = self.local(s[4], s[5])
        rot = yaw_matrix(s[3])
        center = ctr @ rot.T + s[:3]
        half = np.einsum("nij,nj->ni", np.abs(np.matmul(rot, mat)), self.ab[:, 3:])
        reserve = 2 * math.sin(self.limits.tilt_max / 2) * rad
        boundary = np.minimum(center - half - self.lo, self.hi - center - half).min(axis=1)
        near = boundary.copy()
        near_name = np.full(len(near), "outside_observed_map", dtype=object)
        if len(self.eg):
            gap = np.abs(center[:, None, :] - self.ec) - half[:, None, :] - self.eh
            low = np.linalg.norm(np.maximum(gap, 0), axis=2) + np.minimum(np.max(gap, axis=2), 0)
            if exact:
                # Run the narrow phase only for pairs capable of changing the group minimum.
                self.place(self.w, s)
                mujoco.mj_kinematics(self.m, self.w)
                for sel in (self.groups, ~self.groups):
                    ix = np.flatnonzero(sel)
                    best = float(np.min((near - reserve)[ix]))
                    lower = low[ix] - reserve[ix, None]
                    # In penetration the AABB overlap is only a broad-phase flag, not a depth bound.
                    order_bound = np.where(low[ix] <= 0, -np.inf, lower)
                    for flat in np.argsort(order_bound, axis=None):
                        ii, j = np.unravel_index(int(flat), order_bound.shape)
                        if order_bound[ii, j] >= best:
                            break
                        i = ix[ii]
                        dist = self.component_map_clearance(
                            i, j, low[i, j], max(0.05, best + reserve[i] + 0.002)
                        )
                        if dist < near[i]:
                            near[i] = dist
                            near_name[i] = self.boxes[j][0]
                        best = min(best, near[i] - reserve[i])
            else:
                j = np.argmin(low, axis=1)
                v = low[np.arange(len(j)), j]
                use = v < near
                near[use] = v[use]
                near_name[use] = np.array([self.boxes[idx][0] for idx in j[use]], dtype=object)
        eff = near - reserve
        ii = int(np.flatnonzero(self.groups)[np.argmin(eff[self.groups])])
        jj = int(np.flatnonzero(~self.groups)[np.argmin(eff[~self.groups])])
        ib = ii if eff[ii] - self.limits.body_margin < eff[jj] - self.limits.arm_margin else jj
        reason = (
            "self_collision"
            if hit
            else (
                ("body_environment" if ib == ii else "arm_environment")
                if eff[ib] < (self.limits.body_margin if ib == ii else self.limits.arm_margin)
                else None
            )
        )
        r = dict(
            body=float(eff[ii]),
            arm=float(eff[jj]),
            raw_body=float(near[ii]),
            raw_arm=float(near[jj]),
            body_pair=[self.names[self.ids[ii]], str(near_name[ii])],
            arm_pair=[self.names[self.ids[jj]], str(near_name[jj])],
            tilt_reserve_body=float(reserve[ii]),
            tilt_reserve_arm=float(reserve[jj]),
            reason=reason,
            pair=list(hit[1:]) if hit else [self.names[self.ids[ib]], str(near_name[ib])],
            self_contact=list(hit) if hit else None,
        )
        if not exact:
            if len(self.fastcache) >= 8192:
                self.fastcache.popitem(last=False)
            self.fastcache[k] = r
        return r

    def search_clearances(self, s):
        r = self.state(s)
        return r["body"], (-1.0 if r["reason"] in ("self_collision", "command_range") else r["arm"])

    def edge(self, a, b):
        self.calls["seed_edges"] += 1
        a = np.asarray(a)
        delta = np.asarray(b) - a
        delta = delta.copy()
        delta[3] = wrap(delta[3])
        delta[5] = wrap(delta[5])
        n = max(
            1,
            int(
                math.ceil(
                    max(
                        np.linalg.norm(delta[:3]) / 0.035,
                        abs(delta[3]) / 0.13,
                        abs(delta[4]) / 0.095,
                        shape_span(a, b) / 0.13,
                    )
                )
            ),
        )
        return all(not self.state(a + u * delta)["reason"] for u in np.linspace(0, 1, n + 1))

    def proof(self, spline, dt=0.04, dp=0.01, da=math.radians(0.5)):
        times = sampled_times(spline, dt, dp, da)
        states = spline.evaluate(times, 0)
        first = None
        body = arm = float("inf")
        count = 0
        bp = ap = None
        for i, (t, s) in enumerate(zip(times, states)):
            q = self.state(s, True)
            count += 1
            if q["body"] < body:
                body = q["body"]
                bp = q.get("body_pair")
            if q["arm"] < arm:
                arm = q["arm"]
                ap = q.get("arm_pair")
            if q["reason"]:
                first = dict(index=i, time_s=float(t), state=s.tolist(), **q)
                break
        return dict(
            accepted=first is None,
            first=first,
            samples=count,
            body_min=body,
            arm_min=arm,
            body_pair=bp,
            arm_pair=ap,
            map_hash=self.map_hash,
            platform_id=PLATFORM,
            angular_step_deg=math.degrees(da),
            translation_step_m=dp,
            max_time_step_s=dt,
            self_gap_m=SELF_GAP,
            tilt_bound_rad=self.limits.tilt_max,
        )
