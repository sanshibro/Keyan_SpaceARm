"""Deterministic vertex clustering for SDK visual mesh; original kept verbatim."""
from pathlib import Path
import struct,json,hashlib
import numpy as np
p=Path(__file__).resolve().parent/'vendor/qingyu_continuum_arm/meshes/A01Base.STL'
b=p.read_bytes();n=struct.unpack_from('<I',b,80)[0]
a=np.ndarray(n,dtype=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')]),buffer=b,offset=84)['v'].reshape(-1,3)
# Original millimetres; 0.3mm clustering. Deterministic centroid per cluster.
_,idx,inv=np.unique(np.round(a/.3).astype(np.int64),axis=0,return_index=True,return_inverse=True)
v=np.zeros((len(idx),3));np.add.at(v,inv,a);v/=np.bincount(inv)[:,None]
f=inv.reshape(-1,3);f=f[(f[:,0]!=f[:,1])&(f[:,1]!=f[:,2])&(f[:,0]!=f[:,2])]
_,keep=np.unique(np.sort(f,axis=1),axis=0,return_index=True);f=f[np.sort(keep)]
out=p.with_name('A01Base_visual.obj')
with out.open('w') as h:
 for x in v:h.write('v '+' '.join(map(str,x))+'\n')
 for x in f+1:h.write('f '+' '.join(map(str,x))+'\n')
report={'source_sha256':hashlib.sha256(b).hexdigest(),'original_faces':n,'visual_faces':len(f),'cluster_mm':.3,'max_vertex_displacement_mm':float(np.linalg.norm(a-v[inv],axis=1).max()),'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
out.with_suffix('.json').write_text(json.dumps(report,indent=2));print(report)
