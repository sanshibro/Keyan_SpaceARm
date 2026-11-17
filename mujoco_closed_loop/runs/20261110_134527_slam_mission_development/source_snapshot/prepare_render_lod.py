"""Bounded-error visual LOD only. Original vendor CAD and collision proxies unchanged."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent/'vendor/qingyu_continuum_arm/meshes'
report={}
for name,step in [('A01Base_visual',.7)]+[('gripper_'+n,.0005) for n in ['housing','finger0','finger1','finger2']]:
    p=ROOT/(name+'.obj');vertices=[];faces=[]
    for line in p.read_text().splitlines():
        if line.startswith('v '):vertices.append(list(map(float,line.split()[1:4])))
        elif line.startswith('f '):faces.append([int(x.split('/')[0])-1 for x in line.split()[1:4]])
    v=np.array(vertices);f=np.array(faces);_,inv=np.unique(np.round(v/step).astype(np.int64),axis=0,return_inverse=True)
    points=np.zeros((inv.max()+1,3));np.add.at(points,inv,v);points/=np.bincount(inv)[:,None]
    triangles=inv[f];triangles=triangles[(triangles[:,0]!=triangles[:,1])&(triangles[:,1]!=triangles[:,2])&(triangles[:,0]!=triangles[:,2])]
    _,keep=np.unique(np.sort(triangles,axis=1),axis=0,return_index=True);triangles=triangles[np.sort(keep)]
    out=ROOT/(name+'_lod.obj')
    with out.open('w') as h:
        for x in points:h.write('v '+' '.join(map(str,x))+'\n')
        for x in triangles+1:h.write('f '+' '.join(map(str,x))+'\n')
    report[name]={'original_triangles':len(f),'lod_triangles':len(triangles),'max_vertex_error_m':float(np.linalg.norm(v-points[inv],axis=1).max())*(.001 if name.startswith('A01') else 1),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
(ROOT/'render_lod.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
