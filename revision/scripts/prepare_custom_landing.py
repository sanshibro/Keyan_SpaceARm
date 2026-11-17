"""Register the user's native long landing assembly below SUPER rotor sweep.

Only a rigid transform is applied. This is a proposed assembly registration,
not a claim that its mounting holes or load capacity have been measured.
"""
from pathlib import Path
import json,zipfile,hashlib,io
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]

def read_ply(data):
    stream=io.BytesIO(data);lines=[]
    while True:
        line=stream.readline().decode('ascii').strip();lines.append(line)
        if line=='end_header':break
    assert 'format binary_little_endian 1.0' in lines
    nv=int(next(x.split()[-1] for x in lines if x.startswith('element vertex')))
    nf=int(next(x.split()[-1] for x in lines if x.startswith('element face')))
    props=[];active=False
    for x in lines:
        if x.startswith('element '):active=x.startswith('element vertex')
        if active and x.startswith('property '):
            _,kind,name=x.split();assert kind=='float';props.append((name,'<f4'))
    v=np.frombuffer(data,offset=stream.tell(),count=nv,dtype=np.dtype(props))
    vertices=np.column_stack([v[x] for x in ('x','y','z')]).astype(float)
    off=stream.tell()+nv*np.dtype(props).itemsize
    f=np.frombuffer(data,offset=off,count=nf,dtype=np.dtype([('n','u1'),('ids','<i4',(3,))]))
    assert np.all(f['n']==3)
    return vertices,f['ids']

def main():
    manifest=json.loads((ROOT/'revision/assets/cad16_geometry_manifest.json').read_text())
    super_info=json.loads((ROOT/'revision/data/super_metric_geometry.json').read_text())
    plate=manifest['components'][165];lo,hi=np.array(plate['actual_world_bounds_m'])
    bolt_above_plate=max(p['actual_world_bounds_m'][1][1] for p in manifest['components'][151:266])-hi[1]
    target_top=min(r['prop_z_interval_m'][0] for r in super_info['rotors'])-.005-bolt_above_plate
    source_origin=np.array([(lo[0]+hi[0])/2,hi[1],(lo[2]+hi[2])/2])
    R=np.array([[0,0,1],[1,0,0],[0,1,0]],dtype=float)
    translation=np.array([0,0,target_top])-R@source_origin
    out=ROOT/'revision/generated/custom_landing';out.mkdir(parents=True,exist_ok=True)
    parts=[];allv=[]
    with zipfile.ZipFile(ROOT/'revision/assets/cad16_native_meshes.zip') as z:
        for i in range(151,266):
            p=manifest['components'][i];assert p['group']=='airframe'
            raw=z.read(p['file']);assert hashlib.sha256(raw).hexdigest()==p['sha256']
            v,f=read_ply(raw);v=v@R.T+translation;allv.append(v)
            path=out/('landing_%03d.obj'%i)
            with path.open('w') as outmesh:
                np.savetxt(outmesh,v,fmt='v %.9g %.9g %.9g')
                np.savetxt(outmesh,f+1,fmt='f %d %d %d')
            mean=v.mean(0);_,axes=np.linalg.eigh(np.cov((v-mean).T));axes=axes[:,::-1]
            if np.linalg.det(axes)<0:axes[:,-1]*=-1
            local=(v-mean)@axes;l=local.min(0);h=local.max(0)
            center=mean+axes@((l+h)/2);half=(h-l)/2+1e-5
            quat=Rotation.from_matrix(axes).as_quat()[[3,0,1,2]]
            parts.append(dict(index=i,source_part=p['source_part'],file=str(path.relative_to(ROOT)),
                source_sha256=p['sha256'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                color=p['color'],material_role=p['material_role'],
                collision=dict(type='box',center=center.tolist(),half_size=half.tolist(),quat_wxyz=quat.tolist()),
                vertices=len(v),faces=len(f)))
    v=np.vstack(allv)
    result=dict(platform_id=super_info['platform_id'],source='user native 16-series long landing assembly; components 151..265',
        selection_reason='long landing gear visible in user flight photographs',
        source_sha256=hashlib.sha256((ROOT/'revision/assets/cad16_geometry_manifest.json').read_bytes()).hexdigest(),
        transform=dict(rotation=R.tolist(),translation_m=translation.tolist(),uniform_scale=1.),
        registration='highest landing-assembly component 5 mm below lowest rotor sweep; centres aligned',
        mounting_holes_and_load_capacity_verified=False,short_SUPER_feet_replaced=True,
        plate_top_z_m=target_top,bounds_m=[v.min(0).tolist(),v.max(0).tolist()],parts=parts)
    (ROOT/'revision/data/custom_landing_geometry.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ['source','registration','bounds_m','mounting_holes_and_load_capacity_verified']},indent=2))

if __name__=='__main__':main()
