"""Conservative swept-envelope certificate for the offline display scene."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from m0_data import (GATE,QA,Run,body_poses,index,plant,protocol,quat_mat,write_json)
from m0_scene import inspection_scene

def check_static():
    pieces,covers=inspection_scene()
    for part in pieces:
        cover=covers[part.group]
        pts=(part.v-cover['origin'])@np.array(cover['frame'])
        if cover['kind']=='box':
            ok=(pts>=np.array(cover['lo'])-1e-8).all() and (pts<=np.array(cover['hi'])+1e-8).all()
        else:
            rad=np.linalg.norm(pts[:,:2],axis=1)
            face=pts[part.f][:,:,:2].mean(axis=1);length=np.linalg.norm(face,axis=1)
            vec=face/np.maximum(length[:,None],1e-12)
            # Supporting radial half-plane for each triangle, including inner chord sag.
            least=(pts[part.f][:,:,:2]*vec[:,None,:]).sum(-1).min()
            ok=(pts[:,2].min()>=cover['zmin']-1e-8 and pts[:,2].max()<=cover['zmax']+1e-8 and
                rad.max()<=cover['rout']+1e-8 and (cover['rin']==0 or least>=cover['rin']-1e-8))
        if not ok:raise ValueError(f"Static surface escapes cover: {part.name}, {part.group}; {pts.min(0)}, {pts.max(0)}")
    return len(pieces),covers

def init():
    global MODEL,COVERS,GROUPS
    MODEL=plant();_,COVERS=check_static();GROUPS={}
    for b in range(1,MODEL.nbody):
        ids=[i for i in range(MODEL.ngeom) if MODEL.geom_bodyid[i]==b and
             (MODEL.geom_contype[i] or MODEL.geom_conaffinity[i] or
              (MODEL.geom_group[i] in (1,2) and MODEL.geom_rgba[i,3]>.05))]
        if not ids:continue
        centers=[];halves=[];labels=[]
        for i in ids:
            R=quat_mat(MODEL.geom_quat[i]);base=MODEL.geom_pos[i]
            # Use the actual transformed vertices, so bounds do not accumulate
            # two independent axis-aligned approximations of the same mesh.
            if MODEL.geom_type[i]==7:
                mid=MODEL.geom_dataid[i];start=MODEL.mesh_vertadr[mid];n=MODEL.mesh_vertnum[mid]
                v=MODEL.mesh_vert[start:start+n]@R.T+base
                lo=v.min(0);hi=v.max(0)
            else:
                c=base+R@MODEL.geom_aabb[i,:3];h=abs(R)@MODEL.geom_aabb[i,3:]
                lo,hi=c-h,c+h
            centers.append((lo+hi)/2);halves.append((hi-lo)/2);labels.append(MODEL.geom(i).name or str(i))
        GROUPS[b]=np.array(centers),np.array(halves),labels

def lower(lo,hi,c):
    if c['kind']=='box':
        delta=np.maximum(np.array(c['lo'])-hi,lo-np.array(c['hi']))
        return np.max(delta,axis=-1)
    rmin=np.sqrt(np.maximum(np.maximum(lo[:,:,0],-hi[:,:,0]),0)**2+
                 np.maximum(np.maximum(lo[:,:,1],-hi[:,:,1]),0)**2)
    rmax=np.sqrt(np.maximum(abs(lo[:,:,0]),abs(hi[:,:,0]))**2+
                 np.maximum(abs(lo[:,:,1]),abs(hi[:,:,1]))**2)
    axial=np.maximum(c['zmin']-hi[:,:,2],lo[:,:,2]-c['zmax'])
    radial=rmin-c['rout']
    if c['rin']>0:radial=np.maximum(radial,c['rin']-rmax)
    return np.maximum(radial,axial)

def audit_case(case):
    run=Run(case,('physics',))
    assert run.metrics['success'] and run.run['contact_integration_steps']==0
    rot,trans,_=body_poses(MODEL,run.physics['qpos'])
    out={k:dict(lower_bound_m=10.) for k in COVERS}
    orientations={}
    for name,cover in COVERS.items():
        key=tuple(np.array(cover['frame']).flatten())+tuple(cover['origin'])
        orientations.setdefault(key,[]).append(name)
    for body,(centers,ext,labels) in GROUPS.items():
        Q=rot[:,body];x=trans[:,body]
        trace=(Q[:-1]*Q[1:]).sum((1,2))
        angle=np.arccos(np.clip((trace-1)/2,-1,1))
        # A vertex on a slerped body has norm(second derivative) <= r*angle^2.
        # Its distance from the endpoint chord is therefore <= r*angle^2/8.
        sag=angle[:,None]**2*(np.linalg.norm(abs(centers)+ext,axis=1)[None,:])/8+1e-8
        for key,names in orientations.items():
            F=np.array(key[:9]).reshape(3,3);origin=np.array(key[9:])
            rf=F.T@Q;xf=(x-origin)@F
            center=centers[None,:,:]@rf.swapaxes(1,2)+xf[:,None,:]
            half=ext[None,:,:]@np.abs(rf.swapaxes(1,2))
            a=center-half;b=center+half
            l=np.minimum(a[:-1],a[1:])-sag[:,:,None]
            h=np.maximum(b[:-1],b[1:])+sag[:,:,None]
            for name in names:
                gap=lower(l,h,COVERS[name])
                k,i=np.unravel_index(int(np.argmin(gap)),gap.shape)
                d=float(gap[k,i])
                if d<out[name]['lower_bound_m']:
                    out[name]=dict(lower_bound_m=d,release_s=float(run.pt[k]),part=labels[i])
    return dict(case=case,states=len(run.pt),intervals=len(run.pt)-1,inputs=run.inputs,covers=out)

def provenance():
    cases=index()
    results=[];comps=0
    for case,info in cases.items():
        r=Run(case,("physics","control","score-trace"));p=protocol()
        assert r.protocol['gate']==p['gate'] and r.control_summary['goal_xyz']==p['target']
        assert r.control_summary['goal_axis']==p['target_axis_enu']
        assert np.array_equal(r.z['phase'],r.control['phase'][np.searchsorted(r.control['t'],r.t)])
        results.append(dict(case=case,sha256=info['sha256'],success=r.metrics['success'],inputs=r.inputs))
        comps+=int(r.metrics['success'])
    assert len(results)==60 and comps==42
    write_json(QA/'archive-fingerprints.json',dict(source_commit='951c512fbeaf860c5158e18c44aed527dbf7a899',cases=results))
    return [r['case'] for r in results if r['success']]

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--case')
    a=p.parse_args();parts,c=check_static()
    if a.case:
        init();print(audit_case(a.case)['covers'],flush=True);return
    cases=provenance()
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=init) as pool:
        for r in pool.map(audit_case,cases):
            rows.append(r);print('Geometry',len(rows),'/',len(cases),r['case'],flush=True)
    min_by={k:min((r['covers'][k]['lower_bound_m'],r['case'],r['covers'][k]['part']) for r in rows) for k in c}
    cert=dict(physical_model='registered G1; native body kinematics on archived qpos',source_archives=60,
              successful_flights=len(rows),recorded_states=sum(r['states'] for r in rows),
              interpolated_intervals=sum(r['intervals'] for r in rows),static_mesh_parts=parts,
              covers=c,minimum_separating_bounds_m=min_by,cases=rows)
    cert['certified']=all(v[0]>0 for v in min_by.values())
    write_json(QA/'scene-clearance.json',cert)
    print(min_by,flush=True)
    if not cert['certified']:raise SystemExit("Closer inspection required for unresolved bounding volumes.")

if __name__=="__main__":main()
