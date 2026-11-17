"""Split SDK disconnected CAD shells into housing and three finger assemblies."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parent
p=ROOT/'vendor/qingyu_continuum_arm/meshes/gripper.STEP.stl'
b=p.read_bytes();n=struct.unpack_from('<I',b,80)[0]
a=np.ndarray(n,dtype=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')]),buffer=b,offset=84)['v'].reshape(-1,3)
v,inv=np.unique(np.round(a,4),axis=0,return_inverse=True);f=inv.reshape(-1,3)
i=f.ravel();j=np.roll(f,1,axis=1).ravel();_,labels=connected_components(coo_matrix((np.ones(len(i)),(i,j)),shape=(len(v),len(v))),directed=False)
# Shell IDs from canonical lexicographic vertex ordering; source hash pinned.
assert hashlib.sha256(b).hexdigest()=='5ea0d7417e19e2097d885454e670f62ea7d81c633d4ffab83ba4edeaf11829b1'
groups={'housing':[2,4,6],'finger0':[9,10,11],'finger1':[0,1,3],'finger2':[5,7,8]}
report={'source_sha256':hashlib.sha256(b).hexdigest(),'method':'connected CAD shells; three groups by radial finger; approximate inward slides, not linkage reconstruction','parts':{}}
for name,ids in groups.items():
 faces=f[np.isin(labels[f[:,0]],ids)];selected=np.unique(faces);mapping=np.full(len(v),-1);mapping[selected]=np.arange(len(selected))
 # SDK Rz(pi) Rx(pi/2), recentered around the three-finger axis; metre output.
 points=v[selected][:,[0,2,1]]*np.array([-1,1,1])*.001+[-.0021225,-.0231,.061]
 out=p.with_name('gripper_'+name+'.obj')
 with out.open('w') as h:
  for x in points:h.write('v '+' '.join(map(str,x))+'\n')
  for x in mapping[faces]+1:h.write('f '+' '.join(map(str,x))+'\n')
 report['parts'][name]={'shells':ids,'faces':len(faces),'min_m':points.min(0).tolist(),'max_m':points.max(0).tolist(),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
(p.parent/'gripper_split.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
