"""Frozen G1 geometry in the MuJoCo ENU world; all lengths in metres."""
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET
import mujoco
import numpy as np
from model import SimArmBridge
from round_f_plant import LAUNCH, Plant, ROOT

PROTOCOL = ROOT / 'revision/protocols/round_g1.json'
P = json.loads(PROTOCOL.read_text())
DATA = ROOT / 'revision/data/round_g1'
D = ROOT / 'revision/data/round_d/accepted'
SHIFT = np.array([.48, 0., .65])
TARGET = np.array(P['target_tip_ENU_m'])
alpha = np.radians(P['target_tool_tilt_from_down_to_east_deg'])
AXIS = np.array([np.sin(alpha), 0., -np.cos(alpha)])


def boxes(scene):
    if scene == 'wide':
        return [(name, np.asarray(lo) + SHIFT, np.asarray(hi) + SHIFT)
                for name, lo, hi in json.loads((D / 'config.json').read_text())['gate_boxes']]
    g = P['standard_gate']
    x, dx, w = P['gate_center_x'], g['thickness'] / 2, g['half_width_y']
    l, h = g['sill_z'], g['lintel_z']
    yz = [('sill', -1.5, 1.5, 0., l), ('lintel', -1.5, 1.5, h, 3.15),
          ('jamb_negative', -1.5, -w, l, h), ('jamb_positive', w, 1.5, l, h)]
    return [(n, np.array([x-dx, yl, zl]), np.array([x+dx, yh, zh])) for n, yl, yh, zl, zh in yz]


def make_plant(scene='study'):
    def add_gate(root):
        world = root.find('worldbody')
        for item in world.find("body[@name='uav']").iter('geom'):
            if item.get('name', '').startswith(('visual_', 'qingyu_rviz_')) and item.get('contype') == '0':
                item.set('contype', '4')
                item.set('conaffinity', '0')
        for name, lo, hi in boxes(scene):
            ET.SubElement(world, 'geom', name='G1_'+name, type='box', group='4',
                          pos=' '.join(map(str, (lo+hi)/2)), size=' '.join(map(str, (hi-lo)/2)),
                          contype='2', conaffinity='5', rgba='.52 .62 .68 .23')
    return Plant(scene_update=add_gate)


class Geometry:
    """Individual registered CAD/proxy bounding boxes, grouped by rigid base, links and tool."""
    def __init__(self, plant, scene='study'):
        self.m, self.d = plant.m, plant.d
        m = self.m
        self.bridge = SimArmBridge()
        self.ids = np.array([i for i in range(m.ngeom) if m.geom_bodyid[i] != 0 and
            (m.geom_contype[i] or m.geom_conaffinity[i] or
             (m.geom_rgba[i, 3] > .001 and m.geom_group[i] in (1, 2)))])
        rigid = [m.body(n).id for n in ('uav', 'mount')]
        self.cat = np.array(['body' if m.geom_bodyid[i] in rigid else
            ('arm' if m.body(m.geom_bodyid[i]).name.startswith('seg') else 'tool') for i in self.ids])
        self.aabb = m.geom_aabb[self.ids].copy()
        self.gates = boxes(scene)
        self.glo = np.array([lo for _, lo, _ in self.gates])
        self.ghi = np.array([hi for _, _, hi in self.gates])

    def place(self, xyz, theta):
        self.d.qpos[:7] = [*xyz, 1., 0., 0., 0.]
        self.bridge.set_kinematic_shape(self.m, self.d, np.radians(theta), math.pi)
        mujoco.mj_kinematics(self.m, self.d)

    def from_record(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)

    def bounds(self):
        R = self.d.geom_xmat[self.ids].reshape(-1, 3, 3)
        ctr = self.d.geom_xpos[self.ids] + (R @ self.aabb[:, :3, None])[:, :, 0]
        half = (abs(R) @ self.aabb[:, 3:, None])[:, :, 0]
        return ctr - half, ctr + half

    def distances(self, lo, hi):
        # A positive Euclidean box distance proves at least that much convex separation.
        # Zero flags overlapping bounding boxes; physical contact is logged independently.
        g = np.maximum(np.maximum(lo[:, None] - self.ghi, self.glo - hi[:, None]), 0.)
        return np.sqrt(np.einsum('ijk,ijk->ij', g, g))

    def tip(self):
        s = self.d.site('tip')
        return s.xpos.copy(), s.xmat.reshape(3, 3)[:, 2].copy()
