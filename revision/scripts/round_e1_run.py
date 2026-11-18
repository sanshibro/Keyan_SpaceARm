"""E1 integrates only actuator commands, with six fixed-base simulations and native contact auditing."""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from round_e1_model import ROOT, BASE_LENGTH, build, contacts, geometry_controls, neutral
from arm_command import ArmCommand, MujocoArmBackend

PROTOCOL = ROOT / 'revision/protocols/round_e1.json'
P = json.loads(PROTOCOL.read_text())
DEFAULT = ROOT / 'revision/data/round_e1'


def command(t):
    def ease(x): return .5-.5*math.cos(math.pi*max(0.,min(x,1.)))
    if t<5: return 90*ease((t-1)/4)
    if t<7: return 90+28*ease((t-5)/2)
    if t<9: return 118.
    return 118*(1-ease((t-9)/4))


def run(n,dt,out):
    label='n%02d_dt%04dus'%(n,round(dt*1e6))
    model,props=build(n,dt)
    control=geometry_controls(model,n)  # Fresh data below, including in the zero state.
    data=neutral(model)
    backend=MujocoArmBackend(model,data,n)
    segments=np.array([model.body('seg%d'%i).id for i in range(n)])
    mount=model.body('mount').id
    tool=model.body('tool').id
    tip=model.site('tip').id
    start=data.xpos[mount]+data.xmat[mount].reshape(3,3)@np.array([0,0,BASE_LENGTH])
    knots=np.r_[0.,(np.arange(n)+.5)/n,1.]
    fractions=np.linspace(0,1,P['arclength_samples'])
    stride=int(round(P['sample_s']/dt))
    number=int(round(P['duration_s']/dt))
    smooth_data={k:[] for k in ('t','command','net','curvature','backbone','tcp','qpos','qvel','ctrl','ncon')}
    steps=[]; pairs={};peak=0.; limit_steps=0; sat_steps=0; contact_steps=0; commands_limited=0
    peak_torque=0.; max_force=model.actuator_forcerange[backend.ids,1]; max_abs_t=0.
    def read_shape():
        bend=data.qpos[backend.qpos_ids].reshape(-1,2)
        return np.degrees(np.linalg.norm(bend.sum(0))),np.degrees(np.linalg.norm(bend,axis=1).sum())
    def sample(t,net,total):
        mujoco.mj_kinematics(model,data)
        xyz=np.vstack((start,data.xpos[segments],data.xpos[tool]))
        values=dict(t=t,command=command(t),net=net,curvature=total,
                    backbone=np.column_stack([np.interp(fractions,knots,x) for x in xyz.T]),
                    tcp=data.site_xpos[tip].copy(),qpos=data.qpos.copy(),qvel=data.qvel.copy(),
                    ctrl=data.ctrl.copy(),ncon=int(data.ncon))
        for key,val in values.items():smooth_data[key].append(val)
    sample(0.,*read_shape())
    for k in range(number):
        t=k*dt
        issued=backend.apply(ArmCommand(k,t,math.radians(command(t)),math.pi))
        commands_limited+=int(issued['theta_limited'])
        mujoco.mj_step(model,data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or abs(data.time-(k+1)*dt)>1e-7:
            raise ArithmeticError('Non-finite state or unexpected clock reset at %s %d'%(label,k))
        net,total=read_shape()
        if total>peak: peak,max_abs_t=total,data.time
        limit_steps+=int(total>120.)
        torque=np.abs(data.actuator_force[backend.ids])
        saturation=bool(np.any(torque>=max_force-1e-7))
        sat_steps+=int(saturation)
        peak_torque=max(float(torque.max()),peak_torque)
        # mj_step's contact/force fields are those used by the completed step.
        contact_steps+=int(data.ncon>0)
        if data.ncon:
            for a,b,dist in contacts(model,data):
                key=' / '.join(sorted((a,b)))
                p=pairs.setdefault(key,dict(first_step_start_s=t,minimum_m=dist,occurrences=0))
                p['minimum_m']=min(p['minimum_m'],dist);p['occurrences']+=1
        steps.append([t+dt,net,total,torque.max(),data.ncon,saturation])
        if (k+1)%stride==0: sample((k+1)*dt,net,total)
    smooth_data.update(steps=steps,fractions=fractions,base_world=start,
                       step_columns=['t_end_s','net_deg','total_deg','max_abs_arm_torque_Nm','ncon_used','torque_saturated'])
    np.savez_compressed(str(out/(label+'.npz')),**smooth_data)
    row=dict(label=label,N=n,dt_s=dt,steps=number,samples=len(smooth_data['t']),
             peak_raw_curvature_deg=float(peak),peak_time_s=float(max_abs_t),over120_steps=limit_steps,
             max_arm_torque_Nm=peak_torque,saturated_steps=sat_steps,contact_steps=contact_steps,
             contact_pairs=pairs,command_limited_steps=commands_limited,solver_warnings=int(sum(data.warning.number)),
             invariants=props,collision_controls=control)
    (out/(label+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    print('%s | peak %.3f deg | contacts %d | saturated %d'%(label,peak,contact_steps,sat_steps),flush=True)
    return row


def sdk_evidence():
    base=Path('/opt/qingyu_api_ros1')
    paths=['include/qingyu_api_sdk/client.hpp','share/qingyu_api_sdk/examples/cli.cpp',
           'share/qingyu_api_sdk/README.md','share/qingyu_api_sdk/USAGE.md',
           'share/qingyu_api/srv/ConfigReadOne.srv','share/qingyu_api/srv/ConfigWriteOne.srv',
           'share/qingyu_api_sdk/package.xml','lib/continuum_arm/continuum_arm_slider_gui.py']
    files={s:hashlib.sha256((base/s).read_bytes()).hexdigest() for s in paths if (base/s).is_file()}
    return dict(local_prefix=str(base),source_file_sha256=files,
                config_key='uint8_t param_id',supported_values=['Float','Int32','UInt8'],
                write_verification='Installed cli.cpp calls readConfig and compares the returned typed value after writeConfig.',
                joint_angles='SDK joint commands and isolated RViz adapter use degrees; these fields represent bend and plane.',
                flexibility_ratio={'parameter_id':None,'specific_type':None,'unit':None,'torque_or_stiffness_mapping':None,
                                   'finding':'No named flexibility-ratio register or calibrated mapping identified in the inspected files. Device settings and serial transport were not accessed.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT)
    out=parser.parse_args().out
    out.mkdir(parents=True,exist_ok=False)
    (out/'provenance.json').write_text(json.dumps(dict(protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)).decode().strip(),
        mujoco=mujoco.__version__,numpy=np.__version__,sdk=sdk_evidence()),indent=2)+'\n')
    for n in P['segments']:
        for dt in P['step_s']:run(n,dt,out)
