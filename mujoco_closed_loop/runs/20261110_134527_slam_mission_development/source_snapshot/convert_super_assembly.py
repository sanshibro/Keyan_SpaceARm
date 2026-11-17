"""Extract the actual SUPER simulation assembly from its archived Collada file."""
from pathlib import Path
from xml.etree import ElementTree as ET
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'vendor/SUPER/mars_uav_sim/perfect_drone_sim/meshes/yunque-M.dae'
DEST=ROOT/'vendor/super_converted/full_assembly'
NS={'c':'http://www.collada.org/2005/11/COLLADASchema'}

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    root=ET.parse(SOURCE).getroot()
    mesh=root.find('.//c:geometry/c:mesh',NS)
    source=mesh.find("c:source[@id='yunque-M-mesh-positions']/c:float_array",NS)
    vertices=np.fromstring(source.text,sep=' ').reshape(-1,3)
    node=next(n for n in root.findall('.//c:visual_scene/c:node',NS) if n.get('name')=='yunque-M')
    transform=np.fromstring(node.find('c:matrix',NS).text,sep=' ').reshape(4,4)
    vertices=vertices@transform[:3,:3].T+transform[:3,3]
    report={'source':str(SOURCE.relative_to(ROOT)),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'transform':transform.tolist(),'parts':{}}
    for group in mesh.findall('c:triangles',NS):
        material=group.get('material').replace('-material','').replace('-','_')
        stride=1+max(int(i.get('offset')) for i in group.findall('c:input',NS))
        vertex_offset=next(int(i.get('offset')) for i in group.findall('c:input',NS) if i.get('semantic')=='VERTEX')
        raw=np.fromstring(group.find('c:p',NS).text,sep=' ',dtype=np.int32).reshape(-1,3,stride)
        faces=raw[:,:,vertex_offset]
        used, inverse=np.unique(faces,return_inverse=True)
        local=vertices[used];faces=inverse.reshape(-1,3)+1
        path=DEST/(material+'.obj')
        with path.open('w') as f:
            f.write('# SUPER yunque-M assembled mesh; exact source/material indices\n')
            for x,y,z in local:f.write(f'v {x:.7f} {y:.7f} {z:.7f}\n')
            for a,b,c in faces:f.write(f'f {a} {b} {c}\n')
        report['parts'][material]={'vertices':len(local),'triangles':len(faces),'bounds_min':local.min(0).tolist(),'bounds_max':local.max(0).tolist(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        print(material,len(local),len(faces),flush=True)
    (DEST/'provenance.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
