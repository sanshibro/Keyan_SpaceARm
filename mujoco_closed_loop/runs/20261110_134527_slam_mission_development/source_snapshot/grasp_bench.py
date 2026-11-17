"""Contact-only grasp development bench; driven fixture, not flight acceptance."""
import csv,json,time,hashlib
from pathlib import Path
from xml.etree.ElementTree import fromstring,SubElement as S,tostring
import numpy as np,mujoco
from model import model_xml
ROOT=Path(__file__).resolve().parent

def run():
    out=ROOT/'runs'/(time.strftime('%Y%m%d_%H%M%S')+'_grasp_bench');out.mkdir(parents=True)
    root=fromstring(model_xml());u=root.find("worldbody/body[@name='uav']")
    S(u,'joint',name='fixture_x',type='slide',axis='1 0 0',damping='5')
    S(u,'joint',name='fixture_z',type='slide',axis='0 0 1',damping='5')
    act=root.find('actuator')
    for name in ['fixture_x','fixture_z']:S(act,'position',name=name,joint=name,kp='5000',kv='100',forcerange='-100 100')
    world=root.find('worldbody');tabletop=.978
    S(world,'geom',name='support',type='box',pos=f'0 0 {tabletop-.025}',size='.06 .06 .025',rgba='.4 .3 .2 1')
    obj=S(world,'body',name='object',pos=f'0 0 {tabletop+.02}');S(obj,'freejoint',name='object_free')
    S(obj,'geom',name='target',type='box',size='.012 .012 .02',mass='.04',friction='1 .01 .001',rgba='.9 .1 .05 1')
    xml=tostring(root,encoding='unicode');(out/'model.xml').write_text(xml)
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    body=m.body('object').id;tool=m.body('tool').id;target_id=m.geom('target').id;pads=[m.geom('pad%d'%i).id for i in range(3)];support=m.geom('support').id
    initial=d.xpos[body].copy();rows=[];raw=[];contacts_frames=0;bad=0;max_lift=0.;max_carry=0.;forces=[]
    for step in range(11000):
        t=d.time;close=.024*np.clip((t-1)/1,0,1);lift=.10*np.clip((t-3)/2,0,1);carry=.50*np.clip((t-5)/3,0,1)
        d.ctrl[16:19]=close;d.ctrl[m.actuator('fixture_x').id]=carry;d.ctrl[m.actuator('fixture_z').id]=lift+2.25*9.81/5000
        mujoco.mj_step(m,d);n=0;normal=0.;unexpected=0;touching=set()
        for k in range(d.ncon):
            c=d.contact[k];pair={c.geom1,c.geom2}
            if target_id in pair and any(p in pair for p in pads):
                f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f);normal+=f[0];n+=1;touching.update(p for p in pads if p in pair)
            elif pair!={target_id,support}:unexpected+=1
        contacts_frames+=int(n>=2);bad+=unexpected
        z=d.xpos[body,2]-initial[2];x=d.xpos[body,0]-initial[0];max_lift=max(max_lift,z);max_carry=max(max_carry,x)
        if step%10==0:raw.append(np.r_[d.time,d.qpos.copy(),d.qvel.copy(),d.ctrl.copy()])
        if step%10==0:rows.append([t,close,lift,carry,*d.xpos[body],*d.xpos[tool],n,normal,unexpected,*d.qpos[:2],*d.qpos[18:21]])
        if t>=8:forces.append([len(touching),z,x,np.linalg.norm(d.xpos[body]-d.xpos[tool])])
    with (out/'contact.csv').open('w') as h:
        w=csv.writer(h);w.writerow(['sim_time','grip_command','lift_command','carry_command','object_x','object_y','object_z','tool_x','tool_y','tool_z','finger_object_contacts','normal_force_N','unexpected_contacts','fixture_x','fixture_z','finger0','finger1','finger2']);w.writerows(rows)
    np.savez_compressed(out/'raw_dynamics.npz',data=np.array(raw),nq=m.nq,nv=m.nv,nu=m.nu)
    (out/'source.py').write_text(Path(__file__).read_text())
    final=np.array(forces);result=dict(stage='driven_fixture_contact_grasp_only',full_task_accepted=False,feedback='truth_development',fixture='actuated_xz_test_fixture_not_free_flight',binding_used=False,contacts_frames=contacts_frames,unexpected_contact_count=bad,max_lift_m=float(max_lift),max_carry_m=float(max_carry),final_min_lift_m=float(final[:,1].min()),final_min_carry_m=float(final[:,2].min()),final_min_distinct_fingers=int(final[:,0].min()),object_mass_kg=.04,model_sha256=hashlib.sha256(xml.encode()).hexdigest())
    result['passed']=bool(final[:,0].min()>=3 and final[:,1].min()>=.05 and final[:,2].min()>=.45 and bad==0)
    (out/'manifest.json').write_text(json.dumps(result,indent=2));print(out);print(json.dumps(result))
if __name__=='__main__':run()
