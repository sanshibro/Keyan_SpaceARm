"""Register the native CAD camera/mount relative to the same MID360 dome."""
from pathlib import Path
import json,hashlib,zipfile
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from prepare_custom_landing import read_ply
ROOT=Path(__file__).resolve().parents[2]
native=json.loads((ROOT/'revision/assets/cad16_geometry_manifest.json').read_text())
super_info=json.loads((ROOT/'revision/data/super_metric_geometry.json').read_text())
part=super_info['parts']['Lidar'];v=[];f=[]
for line in (ROOT/part['file']).read_text().splitlines():
    if line.startswith('v '):v.append([float(x) for x in line.split()[1:4]])
    elif line.startswith('f '):f.append([int(x.split('/')[0])-1 for x in line.split()[1:4]])
v=np.array(v);f=np.array(f);ii=f.ravel();jj=np.roll(f,1,axis=1).ravel()
n,labels=connected_components(coo_matrix((np.ones(len(ii)),(ii,jj)),shape=(len(v),len(v))),directed=False)
# Select the isolated upper optical dome, not the material group's overall
# bounds (which also contain electronics lower down in the assembly).
top=[]
for i in range(n):
    p=v[labels==i]
    if len(p)>30 and p[:,2].min()>.098 and p[:,2].max()>.12:top.append(p)
assert len(top)==1
target_dome=(top[0].min(0)+top[0].max(0))/2
source_dome=np.mean(native['components'][100]['actual_world_bounds_m'],axis=0)
R=np.array([[0,0,1],[1,0,0],[0,1,0]],float)
translation=target_dome-R@source_dome
camera_parts=[native['components'][i] for i in range(102,137)]
lo=np.min([p['actual_world_bounds_m'][0] for p in camera_parts],axis=0)
hi=np.max([p['actual_world_bounds_m'][1] for p in camera_parts],axis=0)
source_camera=(lo+hi)/2;camera=R@source_camera+translation
mount=native['components'][98]
with zipfile.ZipFile(ROOT/'revision/assets/cad16_native_meshes.zip') as z:raw=z.read(mount['file'])
assert hashlib.sha256(raw).hexdigest()==mount['sha256']
mv,mf=read_ply(raw);mv=mv@R.T+translation
out=ROOT/'revision/generated/custom_camera';out.mkdir(exist_ok=True,parents=True)
path=out/'mid360_realsense_mount.obj'
with path.open('w') as h:
    for p in mv:h.write('v '+' '.join('%.9g'%x for x in p)+'\n')
    for q in mf+1:h.write('f '+' '.join(map(str,q))+'\n')
result=dict(source='native modular CAD components 98, 100, 102..136',
    source_manifest_sha256=hashlib.sha256((ROOT/'revision/assets/cad16_geometry_manifest.json').read_bytes()).hexdigest(),
    source_dome_center=source_dome.tolist(),target_dome_center=target_dome.tolist(),
    source_camera_center=source_camera.tolist(),source_relative_camera= (source_camera-source_dome).tolist(),
    rotation=R.tolist(),translation=translation.tolist(),body_camera_center=camera.tolist(),
    placement='preserve native camera-to-MID360 translation; rigid registration, unit scale',
    camera_half_size=((hi-lo)[[2,0,1]]/2).tolist(),
    mounting_plate_file=str(path.relative_to(ROOT)),mount_source_sha256=mount['sha256'],
    mount_bounds=[mv.min(0).tolist(),mv.max(0).tolist()],
    mechanical_status='native sensor mounting plate registered to SUPER; attachment holes require final integration check')
(ROOT/'revision/data/body_camera_registration.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['body_camera_center','placement','source_relative_camera']},indent=2))
