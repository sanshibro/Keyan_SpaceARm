"""Score reconstructed arm geometry against independent marker and full-plant evidence."""
import argparse
import json
import math
import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from gate_shape_timing import enu_body_rotation
from round_g1_model import Geometry,make_plant
from round_shape_pixels import OUTPUT,EVIDENCE,SPEC,sample_rows
from round_shape_sensors import Channels,NODES,features,predict,mocap_bend
from round_shape_fuse import fuse

HEIGHTS=np.linspace(0,.16,33)
METHODS=['command','strain','vision','fusion','joint_pcc']
CONDITIONS=['natural','blackout']


def clearance(geo):
    lo,hi=geo.bounds()
    standard=geo.distances(lo,hi)
    others=standard[:,1:].min(axis=1)
    sill_lo,sill_hi=geo.glo[0],geo.ghi[0]
    gxy=np.maximum(np.maximum(lo[:,:2]-sill_hi[:2],sill_lo[:2]-hi[:,:2]),0.)
    gz=np.maximum(np.maximum(lo[:,2,None]-(sill_hi[2]+HEIGHTS),sill_lo[2]-hi[:,2,None]),0.)
    values=np.minimum(np.sqrt(np.sum(gxy*gxy,axis=1)[:,None]+gz*gz),others[:,None])
    assert np.allclose(values[:,0],standard.min(1),atol=1e-9)
    return np.stack([values[geo.cat=='body'].min(0),values[geo.cat!='body'].min(0)])


def geometry_points(geo):
    ids=[geo.m.body(n).id for n in NODES[1:-1]]
    points=np.vstack([geo.d.xpos[ids],geo.d.site('tip').xpos])
    mount=geo.d.body('mount')
    return points,(points-mount.xpos)@mount.xmat.reshape(3,3)


def place_only_estimated(geo,xyz,R,b):
    mag=np.linalg.norm(b)
    if not np.isfinite(mag) or mag>np.deg2rad(120):return False
    geo.d.qpos[:]=geo.m.qpos0
    geo.d.qpos[:3]=xyz
    quat=Rotation.from_matrix(R).as_quat()
    geo.d.qpos[3:7]=quat[[3,0,1,2]]
    geo.bridge.set_kinematic_shape(geo.m,geo.d,float(mag),float(np.arctan2(b[1],b[0])))
    mujoco.mj_kinematics(geo.m,geo.d)
    return True


def gather(folder,plant):
    rows,raw,idx=sample_rows(folder)
    meta=json.loads((folder/'run.json').read_bytes());release=meta['release_us']
    us=raw['us'];chosen=(us>=release)&(us<=release+46.3e6)
    ii=np.flatnonzero(chosen)
    t=(us[ii]-release)/1e6
    qpos=raw['qpos'][ii]
    gt=Geometry(plant);model=plant.m
    neutral=mujoco.MjData(model);mujoco.mj_forward(model,neutral)
    gauges=Channels(model,neutral,'flight-'+folder.name)
    adc=[];markers=[];tcp=[];body_nodes=[]
    for now,q in zip(t,qpos):
        gt.from_record(q)
        count,marks=gauges.read(gt.d,now)
        adc.append(count);markers.append(marks)
        pts,_=geometry_points(gt)
        body_nodes.append(pts[:-1]);tcp.append(pts[-1])
    marker=np.array(markers)
    b_true=np.stack([mocap_bend(a) for a in marker])
    hil=pd.read_csv(folder/'hil.csv',usecols=['us','issued_deg'])
    k=np.searchsorted(hil.us.to_numpy(),us[ii])
    np.testing.assert_array_equal(hil.us.to_numpy()[k],us[ii])
    cmd=np.column_stack([-np.deg2rad(hil.issued_deg.to_numpy()[k]),np.zeros(len(t))])
    joint=qpos[:,plant.arm.qpos_ids].reshape(-1,8,2).sum(1)
    js=np.searchsorted(ii,idx)
    assert np.array_equal(ii[js],idx)
    return rows,js,t,np.array(adc),marker,b_true,cmd,joint,qpos,np.array(body_nodes),np.array(tcp)


def run(case,model):
    folder=EVIDENCE/'runs'/case
    target=OUTPUT/'runs'/case;target.mkdir(parents=True,exist_ok=True)
    plant=make_plant()
    rows,j,t,adc,markers,truth,cmd,joint,qpos,nodes,tcp=gather(folder,plant)
    strain,tare=features(adc,t)
    sb=predict(model,strain)
    packets=np.load(str(OUTPUT/'frames'/case/'packets.npz'))['rows']
    np.testing.assert_allclose(t[j],packets[:,1],atol=1e-8)
    np.savez_compressed(target/'observations.npz',t=t,strain_adc=adc,tare=tare,wrist_packets=packets,
                        px4_xyz=[r['position'] for r in rows],px4_rpy=[r['rpy_frd_ned'] for r in rows],
                        issued_b=cmd)
    np.savez_compressed(target/'scoring_truth.npz',t=t,marker_world=markers,mocap_b=truth,interior_nodes=nodes,tcp=tcp)
    estgeom=Geometry(make_plant());truthgeom=Geometry(plant)
    truthclear=[];truepoints=[];localpoints=[]
    for index in j:
        truthgeom.from_record(qpos[index])
        truthclear.append(clearance(truthgeom))
        a,b=geometry_points(truthgeom);truepoints.append(a);localpoints.append(b)
    estimated={}
    for condition in CONDITIONS:
        e=fuse(t,sb,packets,model['strain_phi_sigma_deg'],None if condition=='natural' else (31.,35.))
        bvec=np.stack([cmd,sb,e['vision'],e['fusion'],joint])
        n,m=len(METHODS),len(j)
        world=np.full((n,m,8,3),np.nan);local=world.copy()
        cl=np.full((n,m,2,len(HEIGHTS)),np.nan)
        for h,name in enumerate(METHODS):
            if condition=='blackout' and name not in ('vision','fusion'):
                world[h],local[h],cl[h]=[a[h] for a in estimated['natural']]
                continue
            for k,(event,index) in enumerate(zip(rows,j)):
                b=bvec[h,index]
                if place_only_estimated(estgeom,np.array(event['position']),enu_body_rotation(event['rpy_frd_ned']),b):
                    world[h,k],local[h,k]=geometry_points(estgeom)
                    cl[h,k]=clearance(estgeom)
        estimated[condition]=(world,local,cl)
        np.savez_compressed(target/(condition+'.npz'),t=t[j],state_index=j,methods=METHODS,heights=HEIGHTS,
                            bend=bvec[:,j],world=world,local=local,clear=cl,
                            reference_b=truth[j],reference_world=truepoints,reference_local=localpoints,reference_clear=truthclear,
                            w_theta=e['w_theta'],w_phi=e['w_phi'],time_50=t,fresh=e['fresh'],vision_age=e['vision_age_s'],
                            theta_phi_b_50=bvec[:4],reference_b_50=truth)
    print(case,'gauges',len(t),'camera',len(j),'geometry per condition',len(HEIGHTS)*len(j),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int)
    args=ap.parse_args()
    model=json.loads((OUTPUT/'bench/predictor.json').read_bytes())
    for s in SPEC['cases']['calibration']+SPEC['cases']['held_out']:
        if args.seed not in (None,s):continue
        for method in SPEC['cases']['methods']:run(f'{s}-{method}',model)
