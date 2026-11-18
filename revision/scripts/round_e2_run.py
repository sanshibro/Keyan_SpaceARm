"""Fixed-airframe sensitivity study. Inputs are virtual actuator coefficients and a rigid tip ballast."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import mujoco
import numpy as np

from round_e1_model import ROOT, SimArmBridge, build, contacts, mass_properties, neutral
from round_e1_run import command
from arm_command import ArmCommand, MujocoArmBackend

PROTOCOL = ROOT / 'revision/protocols/round_e2.json'
P = json.loads(PROTOCOL.read_text())
DEFAULT = ROOT / 'revision/data/round_e2'
F = P['factors']


def save_json(path,value):
    path=Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,mode='w',delete=False) as tmp:
        json.dump(value,tmp,indent=2,allow_nan=False);tmp.write('\n')
    os.replace(tmp.name,str(path))


def cases():
    preferred=[(1.,1.,0),(1.,1.,100),(.75,1.,0),(1.,1.4,0),(1.25,1.,0),(1.,.6,0)]
    grid=itertools.product(F['stiffness_scale'],F['drive_damping_scale'],F['added_payload_g'])
    for k,c,g in dict.fromkeys(preferred+list(grid)):yield (8,.0005,k,c,g)
    for s in P['crosschecks']:yield (s['N'],s['dt_s'],s['stiffness_scale'],s['drive_damping_scale'],s['added_payload_g'])


def tag(n,dt,k,c,g):
    return 'n%02d_t%04d_k%03d_d%03d_m%03d'%(n,round(dt*1e6),round(100*k),round(100*c),g)


def shift(r):
    return (r@r)*np.eye(3)-np.outer(r,r)


def load_check_pose(m,n):
    d=neutral(m)
    SimArmBridge(n).set_kinematic_shape(m,d,np.pi/2,np.pi)
    mujoco.mj_forward(m,d)
    jac=np.zeros((3,m.nv))
    mujoco.mj_jacSite(m,d,jac,None,m.site('tip').id)
    return d.qfrc_bias.copy(),jac


def add_ballast(m,n,grams):
    """Merge tool inertia with a solid sphere centered at the native TCP. No new contact geometry."""
    mass=grams/1000.
    rest=neutral(m); tool=m.body('tool').id; tip=m.site('tip').id
    assert m.site_bodyid[tip]==tool
    ids=np.arange(1,m.nbody);before=mass_properties(m,rest,ids)
    tool_mass=float(m.body_subtreemass[tool]);tcp_world=rest.site_xpos[tip].copy()
    old_bias,jac=load_check_pose(m,n)
    radius=.015
    if mass:
        old=float(m.body_mass[tool]);a=m.body_ipos[tool].copy();b=m.site_pos[tip].copy()
        new=old+mass; center=(old*a+mass*b)/new
        flat=np.empty(9);mujoco.mju_quat2Mat(flat,m.body_iquat[tool]);R=flat.reshape(3,3)
        I=R@np.diag(m.body_inertia[tool])@R.T+old*shift(a-center)
        I+=np.eye(3)*(.4*mass*radius**2)+mass*shift(b-center)
        moments,axes=np.linalg.eigh(I)
        if np.linalg.det(axes)<0:axes[:,0]*=-1
        quat=np.empty(4);mujoco.mju_mat2Quat(quat,np.ascontiguousarray(axes).ravel())
        m.body_mass[tool]=new;m.body_ipos[tool]=center;m.body_inertia[tool]=moments;m.body_iquat[tool]=quat
        mujoco.mj_setConst(m,mujoco.MjData(m))
    rest=neutral(m);after=mass_properties(m,rest,ids)
    orig_center=np.asarray(before['com']);orig_mass=before['mass']
    new_center=(orig_mass*orig_center+mass*tcp_world)/(orig_mass+mass)
    tensor=np.asarray(before['inertia'])+orig_mass*shift(orig_center-new_center)+mass*shift(tcp_world-new_center)+np.eye(3)*(.4*mass*radius**2)
    np.testing.assert_allclose(after['mass'],orig_mass+mass,atol=2e-12,rtol=0)
    np.testing.assert_allclose(m.body_subtreemass[tool],tool_mass+mass,atol=2e-12,rtol=0)
    np.testing.assert_allclose(after['com'],new_center,atol=2e-12,rtol=0)
    np.testing.assert_allclose(after['inertia'],tensor,atol=2e-12,rtol=0)
    bias,_=load_check_pose(m,n)
    residual=max(abs(bias-old_bias+jac.T@(mass*m.opt.gravity)))
    assert residual<1e-9, residual
    return dict(tool_subtree_kg=float(m.body_subtreemass[tool]),whole_mass_kg=after['mass'],
                added_ballast_kg=mass,ballast_radius_m=radius,tool_local_TCP_m=m.site_pos[tip].tolist(),
                neutral_inertia_kgm2=after['inertia'],gravity_torque_residual_Nm=float(residual))


def simulation(cell,out):
    n,dt,k,c,g=cell;key=tag(*cell)
    m,props=build(n,dt);audit=add_ballast(m,n,g)
    aids=np.array([m.actuator('%s%d'%(axis,i)).id for i in range(n) for axis in ('bx','by')])
    m.actuator_gainprm[aids,0]*=k;m.actuator_biasprm[aids,1]*=k;m.actuator_biasprm[aids,2]*=c
    np.testing.assert_allclose(m.actuator_gainprm[aids,0],-m.actuator_biasprm[aids,1])
    data=neutral(m);backend=MujocoArmBackend(m,data,n)
    tip=m.site('tip').id;tool=m.body('tool').id;maxforce=m.actuator_forcerange[aids,1]
    stride=int(round(.005/dt));state_stride=int(round(.020/dt));count=int(round(14/dt))
    fields={s:[] for s in ('t','command','net','curvature','tcp','end','state_t','qpos','qvel','ctrl')}
    def sample(t,net,total,full):
        mujoco.mj_kinematics(m,data)
        for name,value in [('t',t),('command',command(t)),('net',net),('curvature',total),('tcp',data.site_xpos[tip].copy()),('end',data.xpos[tool].copy())]:
            fields[name].append(value)
        if full:
            for name,value in [('state_t',t),('qpos',data.qpos),('qvel',data.qvel),('ctrl',data.ctrl)]:
                fields[name].append(value.copy() if isinstance(value,np.ndarray) else value)
    sample(0.,0.,0.,True)
    over=sat=con=limited=0;peak=torque=0.;peak_t=0.;pair_log={};finite=True;steps_done=0
    for step in range(count):
        t=step*dt
        issued=backend.apply(ArmCommand(step,t,math.radians(command(t)),math.pi))
        limited+=int(issued['theta_limited'])
        mujoco.mj_step(m,data)
        steps_done=step+1
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or abs(data.time-steps_done*dt)>1e-7:
            finite=False;break
        bend=data.qpos[backend.qpos_ids].reshape(n,2)
        total=float(np.rad2deg(np.linalg.norm(bend,axis=1).sum()))
        if total>peak:peak,peak_t=total,steps_done*dt
        over+=int(total>P['limits']['angle_raw_total_deg'])
        force=np.abs(data.actuator_force[aids]);torque=max(torque,float(max(force)))
        sat+=int(np.any(force>=maxforce-1e-7));con+=int(data.ncon>0)
        if data.ncon:
            for a,b,dist in contacts(m,data):
                pair=' / '.join(sorted([a,b]));entry=pair_log.setdefault(pair,{'first_step_s':t,'min_dist_m':dist,'count':0})
                entry['min_dist_m']=min(entry['min_dist_m'],dist);entry['count']+=1
        if steps_done%stride==0:
            sample(steps_done*dt,float(np.rad2deg(np.linalg.norm(bend.sum(0)))),total,steps_done%state_stride==0)
    row=dict(id=key,N=n,dt_s=dt,k_scale=k,d_scale=c,load_g=g,
        K_Nm_rad=props['drive_stiffness_Nm_rad']*k,D_Nms_rad=props['drive_damping_Nms_rad']*c,
        passive_D_Nms_rad=props['passive_damping_Nms_rad'],torque_limits_Nm=props['torque_limits_Nm'],
        peak_total_deg=peak,peak_at_s=peak_t,peak_torque_Nm=torque,over120_steps=over,saturated_steps=sat,
        native_contact_steps=con,contact_pairs=pair_log,limited_commands=limited,solver_warnings=int(sum(data.warning.number)),
        steps=steps_done,finite=finite,whole_inertia_check=audit)
    row['valid']=bool(finite and steps_done==count and not any([over,sat,con,limited,row['solver_warnings']]))
    with tempfile.NamedTemporaryFile(dir=out,mode='wb',suffix='.npz',delete=False) as tmp:np.savez_compressed(tmp,**fields)
    file=out/(key+'.npz');os.replace(tmp.name,str(file))
    row['state_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
    save_json(out/(key+'.json'),row);return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--resume',action='store_true');args=p.parse_args()
    out=args.out;cells=list(cases());assert len(set(cells))==30
    identity=dict(protocol_hash=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        mujoco=mujoco.__version__,num_cases=len(cells),command='round_e1_run.command',factor_scope=P['bounds_source'])
    out.mkdir(parents=True,exist_ok=args.resume)
    provenance=out/'provenance.json'
    if provenance.exists():
        assert args.resume and identity==json.loads(provenance.read_text())
    else:save_json(provenance,identity)
    finished=[]
    for cell in cells:
        name=tag(*cell);log=out/(name+'.json');state=out/(name+'.npz')
        if log.exists():
            row=json.loads(log.read_text());assert args.resume and row['state_sha256']==hashlib.sha256(state.read_bytes()).hexdigest()
        else:row=simulation(cell,out)
        with np.load(state) as x:
            take=(x['t']>=8.5)&(x['t']<=9.)
            deficit=float(118-np.mean(x['net'][take])) if np.any(take) else None
            finished.append(dict(id=name,k=row['k_scale'],d=row['d_scale'],g=row['load_g'],n=row['N'],dt_ms=1000*row['dt_s'],
                hold_deficit_deg=deficit,peak_total_deg=row['peak_total_deg'],valid=row['valid'],bend=x['net'][::20].round(4).tolist(),t=x['t'][::20].round(3).tolist()))
        save_json(out/'progress.json',dict(done=len(finished),total=len(cells),records=finished))
        print('%02d/%02d %s | deficit %s deg | peak %.3f | valid %s'%(len(finished),len(cells),name,
            'incomplete' if deficit is None else '%.3f'%deficit,row['peak_total_deg'],row['valid']),flush=True)
if __name__=='__main__':main()
