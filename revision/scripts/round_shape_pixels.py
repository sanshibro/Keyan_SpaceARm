"""Replay the wrist RGB-D stream on frozen G1 states. Only pixels and surveyed points enter localization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import mujoco
import numpy as np

from fiducials import add_fiducials, FiducialLocalization
from gate_shape_timing import enu_body_rotation
from estimation import orientation_shape
from round_g1_model import ROOT, make_plant
from sensors import OPTICAL_FROM_GL, RGBDSensor

CONFIG=ROOT/'revision/protocols/round_shape_clearance.json'
SPEC=json.loads(CONFIG.read_bytes())
EVIDENCE=ROOT/'revision/data/round_g1_online'
OUTPUT=Path(os.environ.get('ROUND_SHAPE_OUTPUT',str(ROOT/'revision/data/round_shape_clearance')))


def rig_layout():
    # Floor and the wall beyond the endpoint. Boards never enter the physical gate.
    items=[]
    def place(p,u,v,length):
        items.append((len(items),np.asarray(p,float),np.asarray(u,float),np.asarray(v,float),length))
    for x in np.arange(-.8,3.01,.65):
        for y in (-.7,0.,.7):
            place([x,y,.010],[1,0,0],[0,-1,0],.38)
    for z in (.40,1.05,1.70,2.35,3.0):
        for y in (-.92,-.30,.32,.94):
            place([3.35,y,z],[0,-1,0],[0,0,-1],.38)
    assert len(items)<=50
    return items


def matrix(rotation,pos=None):
    T=np.eye(4)
    T[:3,:3]=rotation
    if pos is not None:T[:3,3]=pos
    return T


def optical(d,m):
    c=m.camera('wrist_rgbd').id
    return matrix(d.cam_xmat[c].reshape(3,3)@OPTICAL_FROM_GL,d.cam_xpos[c])


def calibration(p):
    def body(n):
        b=p.d.body(n)
        return matrix(b.xmat.reshape(3,3),b.xpos)
    return np.linalg.solve(body('uav'),body('mount')),np.linalg.solve(body('tool'),optical(p.d,p.m))


def pixel_rig():
    base=make_plant()
    ub,tc=calibration(base)
    root=ET.fromstring(base.xml)
    for g in root.findall("./worldbody/geom"):
        if g.get('name','').startswith('G1_'):
            g.set('group','0')
            g.set('rgba','.53 .60 .64 1')
    items=rig_layout()
    add_fiducials(root,items)
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    assert model.nq==base.m.nq and [model.joint(i).name for i in range(model.njnt)]==[base.m.joint(i).name for i in range(base.m.njnt)]
    assert model.nbody==base.m.nbody and abs(model.body_subtreemass[model.body('uav').id]-base.mass)<1e-9
    return model,mujoco.MjData(model),ub,tc,items


def sample_rows(folder):
    meta=json.loads((folder/'run.json').read_bytes())
    raw=np.load(str(folder/'physics.npz'))
    all_events=[json.loads(line) for line in (folder/'timing.jsonl').read_text().splitlines()]
    rows=[e for e in all_events if e.get('t') is not None and 14<=e['t']<=46]
    assert len(rows) in (159,160,161) and np.all(np.diff([e['us'] for e in rows])==200000)
    stamps=raw['us']
    def nearest(stamp):
        j=int(np.searchsorted(stamps,stamp))
        return min((max(0,j-1),min(len(stamps)-1,j)),key=lambda i:abs(int(stamps[i])-stamp))
    ids=np.array([nearest(e['us']) for e in rows])
    assert np.max(abs(stamps[ids]-[e['us'] for e in rows]))<=10000
    return rows,raw,ids


def render_case(case,rig,smoke):
    m,d,ub,tc,items=rig
    folder=EVIDENCE/'runs'/case
    rows,raw,ids=sample_rows(folder)
    dest=OUTPUT/('pilot' if smoke else 'frames')/case
    dest.mkdir(parents=True,exist_ok=True)
    loc=FiducialLocalization(np.eye(4),refine=True,robust_depth=True,items=items)
    sensor=RGBDSensor(m,'wrist_rgbd',480,360)
    data=[];pictures=[];depths=[];thumbs=[]
    cv2.setNumThreads(1)
    try:
        indices=[0,50,99,120,155] if smoke else range(len(rows))
        for j in indices:
            obs=rows[j];i=ids[j]
            # The recorded 50 Hz physics frame can follow the PX4 input by 4 ms.
            # Deliver the pixel packet on its actual exposure clock.
            capture_t=obs['t']+(int(raw['us'][i])-int(obs['us']))/1e6
            d.qpos[:]=raw['qpos'][i];d.time=float(capture_t)
            mujoco.mj_forward(m,d)
            packet=sensor.read(d)
            vis=loc.update(packet)
            b=np.full(2,np.nan)
            if vis.valid:
                WU=matrix(enu_body_rotation(obs['rpy_frd_ned']),obs['position'])
                b=orientation_shape(np.linalg.solve(WU@ub,vis.T_WC),tc)
            data.append([j,capture_t,obs['us'],int(raw['us'][i]),int(vis.valid),vis.inliers,
                         vis.reprojection_px,*b])
            thumbs.append(cv2.resize(packet.rgb,(240,180),interpolation=cv2.INTER_AREA))
            if smoke or j in (0,50,99,120,155):
                pictures.append(packet.rgb)
                depths.append(packet.depth_m.astype(np.float32))
                cv2.imwrite(str(dest/('wrist-%03d.png'%j)),cv2.cvtColor(packet.rgb,cv2.COLOR_RGB2BGR))
    finally:
        sensor.close()
    names=np.array(['frame','t','us','physics_us','valid','inliers','reprojection_px','bx_rad','by_rad'])
    np.savez_compressed(dest/'packets.npz',columns=names,rows=np.asarray(data),sample_rgb=pictures,
                        sample_depth_m=depths,K=sensor.K,thumbnails=thumbs,T_UB=ub,T_TC=tc)
    print(case,len(data),int(sum(r[4] for r in data)),'pixel-valid',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--seed',type=int)
    a=ap.parse_args()
    rig=pixel_rig()
    seeds=SPEC['cases']['calibration']+SPEC['cases']['held_out']
    cases=['241118-online_pause'] if a.pilot else [f'{s}-{m}' for s in seeds for m in SPEC['cases']['methods'] if a.seed in (None,s)]
    for case in cases:
        render_case(case,rig,a.pilot)


if __name__=='__main__':main()
