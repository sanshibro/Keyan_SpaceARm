"""Reconstruct a recorded clearance loss and stopping transient on the original M2 model."""
import argparse
from itertools import combinations
import json,math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gate_distance_certificate import boxes
from m2_plant import portable,place


def read(folder,build=portable):
    c=np.load(folder/'control.npz');p=np.load(folder/'physics.npz');run=json.loads((folder/'run.json').read_text())
    t=c['t'];idx=np.searchsorted(p['us'],run['release_us']+np.rint(1e6*t).astype(np.int64))
    physical=p['qpos'][idx];vel=p['qvel'][idx,:3]
    model=build().m;d=mujoco.MjData(model);camera=model.geom('d435').id;sill=model.geom('COL_gate1_sill').id
    def clearance(q):
        d.qpos[:]=q;mujoco.mj_kinematics(model,d)
        return float(boxes(d.geom_xpos[camera],d.geom_xmat[camera].reshape(3,3),model.geom_size[camera],
                           d.geom_xpos[sill],d.geom_xmat[sill].reshape(3,3),model.geom_size[sill],.00001)['distance'])
    def vec(z):return list(map(float,z))
    def event(i):
        actual=physical[i].copy();a=c['plan_q'][i]
        d.qpos[:]=model.qpos0;place(model,d,a[:3],*np.rad2deg(a[[4,5,3]]));nominal=d.qpos.copy()
        changed=[(0,1),(1,2),(2,3),(3,7),(7,len(actual))]
        labels=['along_track_x','lateral_y','altitude_z','attitude','deformed_arm_and_tool']
        gap={}
        for k in range(len(changed)+1):
            for subset in combinations(range(len(changed)),k):
                q=nominal.copy()
                for j in subset:
                    lo,hi=changed[j];q[lo:hi]=actual[lo:hi]
                gap[subset]=1000*clearance(q)
        phi={}
        for k,name in enumerate(labels):
            contribution=0
            for group,g in gap.items():
                if k in group:continue
                w=math.factorial(len(group))*math.factorial(len(changed)-len(group)-1)/math.factorial(len(changed))
                contribution+=w*(gap[tuple(sorted((*group,k)))]-g)
            phi[name]=float(contribution)
        d.qpos[:]=actual;mujoco.mj_kinematics(model,d)
        v=d.body('mount').xmat.reshape(3,3).T@d.body('tool').xmat.reshape(3,3)[:,2]
        theta=np.rad2deg(np.arctan2(np.linalg.norm(v[:2]),v[2]));az=np.rad2deg(np.arctan2(v[1],v[0]))
        actual_R=Rotation.from_quat(actual[[4,5,6,3]])
        return dict(t_s=float(t[i]),planned_xyz_m=vec(a[:3]),actual_xyz_m=vec(actual[:3]),est_xyz_m=vec(c['px4_xyz'][i]),
                    actual_rpy_deg=vec(actual_R.as_euler('xyz',degrees=True)),planned_yaw_deg=float(np.rad2deg(a[3])),
                    planned_theta_deg=float(np.rad2deg(a[4])),true_theta_deg=float(theta),true_phi_deg=float(az),
                    issued_bend_deg=float(np.rad2deg(np.linalg.norm(c['issued_b'][i]))),
                    estimated_bend_deg=float(np.rad2deg(np.linalg.norm(c['est_b'][i]))),
                    plan_dtheta_deg_s=float(np.rad2deg(c['plan_dq'][i,4])),
                    planned_signed_mm=gap[()],real_signed_mm=gap[tuple(range(len(changed)))],shapley_delta_mm=phi,
                    trim_angle_deg=vec(np.rad2deg(c['trim_angle_frd'][i])),trim_accel_m_s2=vec(c['trim_accel_enu'][i]),
                    setpoint_accel_m_s2=vec(c['acc_ff_enu'][i]),setpoint_velocity_m_s=vec(c['ff_enu'][i]))
    cut=np.flatnonzero((c['window_status'][1:]=='stopped')&(c['window_status'][:-1]=='running'))+1
    gates={str(tt):event(np.argmin(abs(t-tt))) for tt in (28.,28.32,28.64,28.86)}
    stops=[]
    for j in cut:
        out=[]
        for dt in [-.5,-.25,-.02,0.,.10,.25,.42,.50,.75,1.0]:
            i=np.argmin(abs(t-(t[j]+dt)))
            out.append(dict(t_s=float(t[i]),dt=float(t[i]-t[j]),true_xyz_m=vec(physical[i,:3]),ref_m=vec(c['ref_world'][i]),
                true_vel_m_s=vec(vel[i]),ref_vel_m_s=vec(c['ff_enu'][i]),ref_acc_m_s2=vec(c['plan_ddq'][i,:3]),
                actual_yaw_deg=float(Rotation.from_quat(physical[i,[4,5,6,3]]).as_euler('xyz',degrees=True)[2])))
        stops.append(out)
    return dict(g1=gates,safe_prefix_stops=stops,end_contacts=run['contact_integration_steps'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('flight',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.write_text(json.dumps(read(a.flight),indent=2)+'\n')
