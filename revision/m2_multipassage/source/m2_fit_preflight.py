"""Check the mounting fit and opposing jaw geometries at all admissible jaw openings."""
import argparse,itertools,json,math
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial import cKDTree
from gate_distance_audit import DistanceAudit
from m2_self_registry import partition

def check(plant,out):
    m=plant.m
    audit=DistanceAudit(m,tolerance_m=1.e-5,strict=False,obstacle_prefix='COL_')
    hazard,fit=partition(m,audit.self_pairs)
    mujoco.mj_kinematics(m,audit.d)
    prox=audit.pair(*fit[0]);pairs=[]
    bodies=[m.body('finger'+str(i)).id for i in range(3)]
    geom=[m.geom('collision_gripper_finger'+str(i)).id for i in range(3)]
    for i,j in itertools.combinations(range(3),2):
        g,h=geom[i],geom[j]
        for k in (i,j):
            assert m.geom_dataid[geom[k]]==m.geom_dataid[m.geom('visual_gripper_finger'+str(k)).id]
        def vertices(k):
            mid=m.geom_dataid[geom[k]];start=m.mesh_vertadr[mid]
            vs=m.mesh_vert[start:start+m.mesh_vertnum[mid]].astype(float)
            return vs@audit.d.geom_xmat[geom[k]].reshape(3,3).T+audit.d.geom_xpos[geom[k]]
        verts=vertices(i);other=vertices(j);tree=cKDTree(other)
        js=[m.joint('grip'+str(k)).id for k in (i,j)]
        assert all(m.jnt_type[k]==mujoco.mjtJoint.mjJNT_SLIDE for k in js)
        directions=audit.d.xaxis[js].copy();rr=m.jnt_range[js]
        corners=[]
        for u,v in itertools.product(*rr):
            audit.d.qpos[m.jnt_qposadr[js]]=[u,v];mujoco.mj_kinematics(m,audit.d)
            hit=audit.pair(g,h)
            corners.append(dict(opening_mm=[u*1000,v*1000],lower_mm=hit['lower']*1000,upper_mm=hit['upper']*1000))
        audit.d.qpos[:]=m.qpos0;mujoco.mj_kinematics(m,audit.d)
        grid=[np.linspace(a,b,13) for a,b in rr];samples=[]
        for u,v in itertools.product(*grid):
            d,k=tree.query(verts+u*directions[0]-v*directions[1],k=1)
            near=int(np.argmin(d));samples.append([float(u),float(v),float(d[near]),near,int(k[near])])
        # Translating two rigid surfaces changes their minimum distance by at most
        # the sum of the two translations. Original triangle vertices supply
        # upper witnesses even for a nonconvex CAD mesh.
        cover=sum((x[1]-x[0])/2 for x in grid)
        upper=max(x[2] for x in samples)+cover
        pairs.append(dict(geoms=[m.geom(g).name,m.geom(h).name],vertex_counts=[len(verts),len(other)],stroke_mm=(1000*rr).tolist(),
            convex_corners=corners,convex_domain_upper_mm=max(0.,max(x['upper_mm'] for x in corners)),
            cad_samples=samples,cad_cover_radius_mm=1000*cover,cad_domain_upper_mm=upper*1000))
    result=dict(registered_pairs=len(audit.self_pairs),nonconnected_pairs=len(hazard),assembly_fit_pairs=len(fit),
                mount_neutral_interval_mm=[1000*prox[k] for k in ('lower','upper')],gripper=pairs)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--retained-robot',action='store_true');args=p.parse_args()
    if args.retained_robot:from m1_replay import build
    else:from m2_plant import build
    r=check(build(),args.output)
    print([(z['geoms'],z['convex_domain_upper_mm'],z['cad_domain_upper_mm']) for z in r['gripper']])
