"""Static native-convex preflight only. This does not execute a flight or planner.

Grid search finds witnesses of feasibility; it does not certify a global optimum.
External pair scans include original physical and CAD-envelope geometry.
"""
import argparse,hashlib,itertools,json,sys,time
from pathlib import Path
import mujoco,numpy as np
from m2_plant import portable,place,option,task_pose,STAGE
from m2_scene import GATES

def original_pairs(m):
    rid=m.body('uav').id;rb=[]
    for body in range(m.nbody):
        k=body
        while k and k!=rid:k=int(m.body_parentid[k])
        if k==rid:rb.append(body)
    robot=np.array([i for i in range(m.ngeom) if m.geom_bodyid[i] in rb and (m.geom_contype[i] or m.geom_conaffinity[i])],int)
    physical=[i for i in robot if m.geom_contype[i]&1 and m.geom_conaffinity[i]&1]
    excluded=set(map(int,m.exclude_signature));explicit={tuple(sorted((int(a),int(b)))) for a,b in zip(m.pair_geom1,m.pair_geom2)}
    parent_filter=not (int(m.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT));pairs=[]
    for a,b in itertools.combinations(physical,2):
        ba,bb=int(m.geom_bodyid[a]),int(m.geom_bodyid[b]);wa,wb=int(m.body_weldid[ba]),int(m.body_weldid[bb])
        if wa==wb:continue
        if (a,b) in explicit:pairs.append((a,b));continue
        pa,pb=int(m.body_weldid[m.body_parentid[wa]]),int(m.body_weldid[m.body_parentid[wb]])
        if parent_filter and wa and wb and (wa==pb or wb==pa):continue
        if (min(ba,bb)<<16)+max(ba,bb) in excluded:continue
        if not ((m.geom_contype[a]&m.geom_conaffinity[b]) or (m.geom_contype[b]&m.geom_conaffinity[a])):continue
        pairs.append((a,b))
    return robot,np.array(pairs,int).reshape(-1,2)

class NativeScan:
    def __init__(self,m,d):
        self.m=m;self.d=d;self.robot,self.self_pairs=original_pairs(m)
        self.obstacles=np.array([i for i in range(m.ngeom) if m.geom(i).name.startswith('COL_')],int)
        self.external=np.array([(a,b) for a in self.robot for b in self.obstacles if
             (m.geom_contype[a]&m.geom_conaffinity[b]) or (m.geom_contype[b]&m.geom_conaffinity[a])],int)
        self.names=[m.geom(i).name or 'geom%d'%i for i in range(m.ngeom)]
    def bounds(self):
        m,d=self.m,self.d;R=d.geom_xmat.reshape(-1,3,3);c=d.geom_xpos+(R@m.geom_aabb[:,:3,None])[:,:,0];h=(abs(R)@m.geom_aabb[:,3:,None])[:,:,0]
        return c-h,c+h
    def scan(self,pairs,upper=2.):
        if not len(pairs):return dict(distance_m=None)
        lo,hi=self.bounds();aa,bb=pairs.T;lb=np.linalg.norm(np.maximum(np.maximum(lo[aa]-hi[bb],lo[bb]-hi[aa]),0),axis=1)
        best=float(upper);record=None;count=0
        for k in np.argsort(lb):
            if lb[k]>max(0,best):break
            a,b=int(aa[k]),int(bb[k]);witness=np.zeros(6);dist=float(mujoco.mj_geomDistance(self.m,self.d,a,b,best if best>0 else .1,witness));count+=1
            if dist<best:
                best=dist;record=dict(distance_m=dist,name_a=self.names[a],name_b=self.names[b],geom_a=a,geom_b=b,
                    body_a=self.m.body(self.m.geom_bodyid[a]).name,body_b=self.m.body(self.m.geom_bodyid[b]).name,
                    point_a=witness[:3].tolist(),point_b=witness[3:].tolist())
        return dict(record or {'distance_m':best},pairs_refined=count)
    def external_score(self):return self.scan(self.external)

def audit(out=STAGE/'qa/static'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);plant=portable();m,d=plant.m,plant.d;scanner=NativeScan(m,d)
    # Root-owned explicit physical-fit classification; every original pair remains audited.
    try:
        from m2_self_registry import partition
        hazard,fit=partition(m,scanner.self_pairs)
    except ImportError:hazard,fit=scanner.self_pairs,np.empty((0,2),int)
    from gate_distance_audit import DistanceAudit
    independent=DistanceAudit(m,tolerance_m=1e-5,strict=True,obstacle_prefix='COL_')
    def serial(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        return value
    def bounded(pair):return {k:serial(v) for k,v in pair.items()}
    results=[];started=time.monotonic();shape_cache={}
    for g in GATES:
        best=None
        phis=[160,180,200] if g['name']!='gate2' else [60,90,120,210,240,270]
        zs=np.linspace(2.23,2.34,12) if g['name']=='gate1' else (np.linspace(1.94,2.02,9) if g['name']=='gate2' else np.linspace(2.20,2.31,12))
        ys=g['y']+np.array([-.06,-.03,0,.03,.06]);thetas=[70,75,80,85]
        for th,ph,y,z in itertools.product(thetas,phis,ys,zs):
            key=(th,ph)
            if key not in shape_cache:
                place(m,d,[0,0,0],th,ph)
                # Native broad search only; every selected feasible witness is
                # subsequently certified by independent convex lower bounds.
                shape_cache[key]=scanner.scan(hazard,upper=.1)['distance_m']>=.016
            if not shape_cache[key]:continue
            xyz=[g['x'],float(y),float(z)];place(m,d,xyz,th,ph)
            e=scanner.external_score()
            if best is None or e['distance_m']>best['external']['distance_m']:
                best=dict(gate=g['name'],xyz=xyz,theta_deg=th,phi_deg=ph,yaw_deg=0.,external=e)
        if best is None:raise RuntimeError('No statically screened shape for '+g['name'])
        place(m,d,best['xyz'],best['theta_deg'],best['phi_deg'])
        independent.d.qpos[:]=d.qpos;mujoco.mj_kinematics(m,independent.d);lo,hi=independent.bounds()
        body=independent._scan('body',independent.external['body'],lo,hi)
        arm=independent._scan('arm_tool',independent.external['arm_tool'],lo,hi)
        best['external_native_search']=best.pop('external')
        best['external']=bounded(body if body['upper']<=arm['upper'] else arm)
        best['external']['distance_m']=best['external']['distance']
        best['external_body']=bounded(body);best['external_arm_tool']=bounded(arm)
        best['self_registered']=bounded(independent._scan('registered',scanner.self_pairs,lo,hi))
        best['self_hazardous']=bounded(independent._scan('hazardous',hazard,lo,hi))
        best['self_fit']=bounded(independent._scan('fit',fit,lo,hi)) if len(fit) else None
        best['external_pass_105mm']=best['external']['lower']>=.105
        results.append(best);print(json.dumps(best),flush=True)
    task=task_pose(plant);place(m,d,task['body'],50,143)
    independent.d.qpos[:]=d.qpos;mujoco.mj_kinematics(m,independent.d);lo,hi=independent.bounds()
    task_parts={k:bounded(independent._scan('task_'+k,v,lo,hi)) for k,v in independent.external.items()}
    task_parts['hazardous']=bounded(independent._scan('task_hazardous',hazard,lo,hi))
    task_parts['fit']=bounded(independent._scan('task_fit',fit,lo,hi)) if len(fit) else None
    result=dict(scene_sha256=plant.scene_digest,mujoco_version=mujoco.__version__,static_only=True,flight_executed=False,
        search='finite native geometry witness search, not a certified global optimum',gate_witnesses=results,
        registered_self_pairs=len(scanner.self_pairs),hazardous_pairs=len(hazard),fit_pairs=len(fit),
        external_geoms=len(scanner.robot),obstacles=len(scanner.obstacles),elapsed_s=time.monotonic()-started,
        task=task,task_native_search=scanner.external_score(),task_certified=task_parts,
        all_external_static_pass=all(r['external_pass_105mm'] for r in results))
    (out/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(out/'gate-witnesses.npz',xyz=[r['xyz'] for r in results],theta_deg=[r['theta_deg'] for r in results],phi_deg=[r['phi_deg'] for r in results])
    registry_path=Path(__file__).with_name('m2_self_registry.py')
    certified=dict(scene_sha256=plant.scene_digest,registry_sha256=hashlib.sha256(registry_path.read_bytes()).hexdigest(),static_only=True,method='independent support-plane and feasible-witness convex bounds, tolerance 10 micrometres',
        gate_witnesses=[dict(gate=r['gate'],external_lower_m=r['external']['lower'],hazard_lower_m=r['self_hazardous']['lower'],
            fit_lower_m=r['self_fit']['lower'] if r['self_fit'] else None,xyz=r['xyz'],theta_deg=r['theta_deg'],phi_deg=r['phi_deg'],
            external_pair=[r['external']['name_a'],r['external']['name_b']],hazard_pair=[r['self_hazardous']['name_a'],r['self_hazardous']['name_b']],
            fit_pair=[r['self_fit']['name_a'],r['self_fit']['name_b']] if r['self_fit'] else None) for r in results],
        external_static_pass=all(r['external_pass_105mm'] for r in results),
        self_static_pass=all(r['self_hazardous']['lower']>=.015 and (r['self_fit'] is None or r['self_fit']['lower']>=0) for r in results),
        task=task,task_bounds=task_parts,detailed_evidence='preflight.json',flight_authorization=False)
    (out/'certified.json').write_text(json.dumps(certified,indent=2),encoding='utf-8')
    return result

def render(out=STAGE/'qa/static'):
    from PIL import Image,ImageDraw,ImageFont
    out=Path(out);record=json.loads((out/'preflight.json').read_text());p=portable();r=mujoco.Renderer(p.m,height=720,width=1280)
    frames=[]
    for row in record['gate_witnesses']:
        place(p.m,p.d,row['xyz'],row['theta_deg'],row['phi_deg']);r.update_scene(p.d,camera='observer_oblique',scene_option=option());pixels=r.render().copy()
        Image.fromarray(pixels).save(out/(row['gate']+'-oblique.png'))
        r.update_scene(p.d,camera='observer_side',scene_option=option());side=r.render().copy();Image.fromarray(side).save(out/(row['gate']+'-side.png'))
        r.update_scene(p.d,camera='wrist_rgbd',scene_option=option());wrist=r.render().copy();Image.fromarray(wrist).save(out/(row['gate']+'-wrist.png'))
        frames.append((pixels,side,row))
    r.close();sheet=Image.new('RGB',(2560,3*780),(255,255,255));draw=ImageDraw.Draw(sheet)
    try:font=ImageFont.truetype('C:/Windows/Fonts/times.ttf',28)
    except OSError:font=ImageFont.load_default()
    for j,(oblique,side,row) in enumerate(frames):
        sheet.paste(Image.fromarray(oblique),(0,j*780+60));sheet.paste(Image.fromarray(side),(1280,j*780+60))
        draw.text((24,j*780+14),'Static %s | shape %.0f / %.0f deg | native external clearance %.1f mm | no flight'%(row['gate'],row['theta_deg'],row['phi_deg'],row['external']['distance_m']*1000),font=font,fill=(25,33,39))
    sheet.save(out/'three-gate-static-review.png')
    # Exact registered wrist view and native metric depth at the terminal pose.
    task=record['task'];place(p.m,p.d,task['body'],task['theta_deg'],task['phi_deg'])
    wrist=mujoco.Renderer(p.m,height=360,width=480)
    wrist.update_scene(p.d,camera='wrist_rgbd',scene_option=option());rgb=wrist.render().copy();Image.fromarray(rgb).save(out/'target-wrist.png')
    wrist.enable_depth_rendering();wrist.update_scene(p.d,camera='wrist_rgbd',scene_option=option());depth=wrist.render().copy();wrist.close()
    Image.fromarray(np.uint16(np.clip(depth*1000,0,65535))).save(out/'target-depth-mm.png')
    from m2_scene import FACE
    cam=p.d.camera('wrist_rgbd');v=cam.xmat.reshape(3,3).T@(FACE-cam.xpos);f=180/np.tan(np.deg2rad(p.m.cam_fovy[p.m.camera('wrist_rgbd').id])/2)
    u=239.5+f*v[0]/-v[2];w=179.5-f*v[1]/-v[2]
    center=float(depth[int(round(w)),int(round(u))]);axis=np.asarray(task['camera_axis']);axis/=np.linalg.norm(axis)
    evidence=dict(scene_sha256=p.scene_digest,face_uv=[float(u),float(w)],optical_depth_m=float(-v[2]),native_depth_at_face_m=center,
       camera_center_depth_m=float(depth[180,240]),valid_depth_fraction=float(np.mean(np.isfinite(depth)&(depth>0)&(depth<10))),
       desired_axis_error_deg=float(np.rad2deg(np.arccos(np.clip(axis@np.asarray(task['axis']),-1,1)))),
       stand_off_pass=bool(.12<=-v[2]<=.18),depth_center_pass=bool(abs(center-(-v[2]))<.003),view_specific_geometry=False)
    (out/'target-optics.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    certified=json.loads((out/'certified.json').read_text());certified['target_optics']=evidence
    (out/'certified.json').write_text(json.dumps(certified,indent=2),encoding='utf-8')
    return evidence
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--render',action='store_true');ap.add_argument('--output',type=Path,default=STAGE/'qa/static');a=ap.parse_args()
    print(json.dumps(render(a.output) if a.render else audit(a.output)))
