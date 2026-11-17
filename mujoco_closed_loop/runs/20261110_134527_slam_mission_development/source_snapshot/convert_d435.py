"""Convert the installed RealSense D435 description mesh to MuJoCo OBJ."""
from pathlib import Path
from xml.etree import ElementTree as E
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/opt/ros/noetic/share/realsense2_description/meshes/d435.dae')
OUT=ROOT/'vendor/realsense_d435';NS={'c':'http://www.collada.org/2005/11/COLLADASchema'}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    root=E.parse(SOURCE).getroot()
    groups={'body':[],'details':[]}
    for geom in root.findall('.//c:geometry',NS):
        mesh=geom.find('c:mesh',NS)
        vertices=np.fromstring(mesh.find('c:source/c:float_array',NS).text,sep=' ').reshape(-1,3)
        faces=np.fromstring(mesh.find('c:triangles/c:p',NS).text,sep=' ',dtype=np.int32).reshape(-1,3)
        groups['body' if geom.get('id')=='ID77' else 'details'].append((vertices,faces))
    provenance={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                'package':'ROS realsense2_description','license':'Apache-2.0 (installed package.xml)',
                'geometry':'all 18 Collada geometries; original metre-scale coordinates','outputs':{}}
    for name,items in groups.items():
        path=OUT/(name+'.obj')
        with path.open('w') as f:
            f.write('# Original D435 Collada geometry from ROS realsense2_description\n')
            offset=0
            for vertices,faces in items:
                for x,y,z in vertices:f.write(f'v {x:.7f} {y:.7f} {z:.7f}\n')
                for a,b,c in faces+offset+1:f.write(f'f {a} {b} {c}\n')
                offset+=len(vertices)
        provenance['outputs'][name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                     'vertices':offset,'triangles':sum(len(f) for _,f in items)}
        print(name,provenance['outputs'][name],flush=True)
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
if __name__=='__main__':main()
