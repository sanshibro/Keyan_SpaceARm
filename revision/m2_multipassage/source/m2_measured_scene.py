"""Figure of one registered M2 flight, reconstructed from native exposure-time states.

The planes cut physical link convexes at fixed world coordinates. Both traces
are drawn in the native depth buffer, so the pipework occludes them correctly.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import mujoco
import numpy as np
from PIL import Image

from m1_sections import Sections
from m2_gate import FORMAL
from m2_plant import build, portable, option
from m2_review_data import FlightRecord, sha
from m2_scene_review import snapshot, restore, camera

STATIONS = (('G1', 'High bulkhead', 1.15), ('G2', 'Duct and cross-pipe', 2.83),
            ('G3', 'Return bulkhead', 3.95))
INK = '#283b3e'
BODY = '#326c77'
TOOL = '#ae673d'


def moments(record):
    move = record.control_summary['motion_start_s']
    x = record.physics['qpos'][:, 0]
    events = []
    for gate, title, plane in STATIONS:
        cross = np.flatnonzero((record.t >= move) & (x >= plane))
        exp = record.exposure(record.t[cross[0]]) if len(cross) else None
        if len(cross) and exp is None:
            raise ValueError('No original exposure at the ' + gate + ' crossing')
        events.append((gate, title, plane, exp))
    dock = record.metrics['motion_to_inspect_s']
    exp = record.exposure(move + dock + .5) if dock is not None else None
    if dock is not None and exp is None:
        raise ValueError('No original exposure in the certified inspection dwell')
    events.append(('E', 'Optical inspection', None, exp))
    return events


def set_pose(plant, q):
    plant.d.qpos[:] = q
    mujoco.mj_kinematics(plant.m, plant.d)
    mujoco.mj_camlight(plant.m, plant.d)


def trace(scene, path, rgba):
    previous = path[0]
    for point in path[1:]:
        if np.linalg.norm(point - previous) < .002:
            continue
        if scene.ngeom >= scene.maxgeom:
            raise ValueError('Ray-traced geometry capacity')
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                            np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, .0032, previous, point)
        scene.ngeom += 1
        previous = point


def panorama(plant, record, events):
    model = plant.m
    model.vis.global_.orthographic = True
    model.vis.global_.fovy = 4.45
    view = camera([2.46, 0., 1.69], 8., 57, -32)
    ghost = []
    opt = option()
    visible = [event[3] for event in events if event[3] is not None]
    if not visible:
        raise ValueError('No flight crossed a registered section')
    with mujoco.Renderer(model, height=900, width=1600, max_geom=6500) as rend:
        for e in visible:
            set_pose(plant, record.physics['qpos'][e['physics_index']])
            rend.update_scene(plant.d, camera=view, scene_option=opt)
            ghost.append([snapshot(g) for g in rend.scene.geoms[:rend.scene.ngeom]
                          if int(g.objtype) == int(mujoco.mjtObj.mjOBJ_GEOM)
                          and g.objid >= 0 and model.geom_bodyid[g.objid] != 0])
        for index, shape in enumerate(ghost[:-1]):
            opacity = .63 + .08 * index
            for saved in shape:
                saved['rgba'][3] *= opacity
                saved['transparent'] = 1
                if rend.scene.ngeom >= rend.scene.maxgeom:
                    raise ValueError('Time-lapse geometry capacity')
                restore(rend.scene.geoms[rend.scene.ngeom], saved)
                rend.scene.ngeom += 1
        selected = np.flatnonzero((record.t >= record.control_summary['motion_start_s']) &
                                  (record.t <= visible[-1]['t'] + 1.e-7))
        step = max(1, int(round(.08 / np.median(np.diff(record.t)))))
        ids = np.unique(np.r_[selected[::step], selected[-1]])
        trace(rend.scene, record.physics['qpos'][ids, :3], [.13, .39, .45, 1.])
        trace(rend.scene, record.physics['tcp'][ids], [.74, .43, .20, 1.])
        return rend.render().copy()


def make(record, out, development=False):
    if record.method != 'coupled' or not record.metrics or record.metrics['development'] != development:
        raise ValueError('Require the audited full method and matching campaign type')
    if not development and record.seed != FORMAL[0]:
        raise ValueError('Use the first preregistered formal seed')
    plant = portable() if os.name == 'nt' else build()
    if record.scene != plant.scene_digest:
        raise ValueError('Native scene fingerprint')
    out.mkdir(parents=True, exist_ok=True)
    events = moments(record)
    wide = panorama(plant, record, events)
    slicer = Sections(plant)
    provenance = dict(scene=record.scene, seed=record.seed, method=record.method,
                      metrics_sha256=sha(record.audit / 'metrics.json'), development=development,
                      abscissae_m=[s[2] for s in STATIONS], events=[])
    plt.rcParams.update({'font.family':'serif','font.serif':['STIX Two Text','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':9,'axes.labelsize':8,'pdf.fonttype':42,
        'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.6,'xtick.labelsize':7,'ytick.labelsize':7})
    fig = plt.figure(figsize=(10.1, 8.3), layout='constrained')
    grid = fig.add_gridspec(3, 12, height_ratios=(3.6, 1.85, 1.62))
    scene_ax = fig.add_subplot(grid[0, :])
    image = wide[65:818, 65:1540]
    scene_ax.imshow(image)
    scene_ax.axis('off')
    scene_ax.set_title('a   Recorded passage through the inspection corridor', loc='left', color=INK)
    for color, label in [(BODY, 'Vehicle origin'), (TOOL, 'Tool centre')]:
        scene_ax.plot([], [], color=color, lw=2.2, label=label)
    scene_ax.legend(loc='upper right', frameon=False, ncol=2, fontsize=8)
    for i, (gate, title, x, exp) in enumerate(events):
        row = dict(station=gate, description=title, plane_m=x, reached=exp is not None)
        col = fig.add_subplot(grid[2, 3*i:3*i+3])
        col.axis('off')
        col.set_title(gate+'   '+title, fontsize=8.5, loc='left')
        sec = fig.add_subplot(grid[1, 4*i:4*i+4]) if i<3 else None
        if sec is not None:
            sec.set(xlim=(-.83,.83),ylim=(1.20,2.83),aspect='equal',xlabel='Lateral position (m)')
            if i==0:
                sec.set_ylabel('Height (m)')
            sec.set_title(chr(ord('b')+i)+'   '+gate+'   $x$ = '+format(x,'.2f')+' m',loc='left',fontsize=9.5)
        if exp is not None:
            n = exp['physics_index']
            with Image.open(exp['rgb_path']) as wr:
                col.imshow(wr.convert('RGB'))
            col.text(.02,-.095,'Exposure %.2f s'%exp['t'],transform=col.transAxes,color=INK,fontsize=7.8)
            row.update(t_s=exp['t'],delivery_s=exp['arrival_t'],frame=exp['frame'],
                       rgb_sha256=exp['rgb_sha256'],depth_sha256=exp['depth_sha256'],
                       pose_sha256=exp['sensor_qpos_sha256'],body=record.physics['qpos'][n,:3].tolist(),
                       tcp=record.physics['tcp'][n].tolist())
        else:
            col.text(.03,.43,'Station not reached',transform=col.transAxes,color='#677477')
        if sec is not None:
            # Draw physical walls even for an unvisited station, with the robot
            # outside the plane. Never move a recorded aircraft into a gate.
            q = record.physics['qpos'][exp['physics_index'] if exp is not None else 0].copy()
            if exp is None:
                q[0] = -5.
            parts = slicer.cut(q,x)
            for name,fill,edge in [('walls','#d6d8d5','#8b9491'),('body','#b7cdd0',BODY),('arm','#e7c9a9',TOOL)]:
                if parts[name]:
                    sec.add_collection(PolyCollection(parts[name],facecolors=fill,edgecolors=edge,linewidths=.47))
            row['cross_section_yz_m'] = parts
        provenance['events'].append(row)
    for ext in ('png','pdf','svg'):
        fig.savefig(out/('measured-corridor.'+ext),dpi=260,facecolor='white')
    plt.close(fig)
    (out/'gate-plane-evidence.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf8')
    return provenance


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('flight',type=Path)
    ap.add_argument('audit',type=Path)
    ap.add_argument('output',type=Path)
    ap.add_argument('--development',action='store_true')
    args = ap.parse_args()
    result = make(FlightRecord(args.flight,args.audit),args.output,args.development)
    print(json.dumps([(r['station'],r['reached'],r.get('t_s')) for r in result['events']]))
