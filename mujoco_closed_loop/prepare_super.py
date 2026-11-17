"""Tessellate pinned official SUPER carbon-fibre STEP parts. SI output."""
from pathlib import Path
import json,hashlib,subprocess
import numpy as np,gmsh
ROOT=Path(__file__).resolve().parent;src=ROOT/'vendor/SUPER-Hardware';out=ROOT/'vendor/super_converted';out.mkdir(exist_ok=True)
report={'source':'https://github.com/hku-mars/SUPER-Hardware','commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),'license':'MIT; original retained in vendor/SUPER-Hardware/LICENSE','converter':'gmsh '+gmsh.__version__,'units':'source mm; output m','parts':{}}
gmsh.initialize();gmsh.option.setNumber('General.Terminal',0)
for name,filename in [('main_board','5mm_main_board.STEP'),('second_board','3mm_sec_board.step'),('battery_board','3mm_bat_board.step'),('foot','4mm-foot.STEP')]:
    gmsh.clear();p=src/'CarbonFiber'/filename;gmsh.model.occ.importShapes(str(p));gmsh.model.occ.synchronize()
    volumes=gmsh.model.getEntities(3);volume=sum(gmsh.model.occ.getMass(*e) for e in volumes)
    centroid=sum(np.array(gmsh.model.occ.getCenterOfMass(*e))*gmsh.model.occ.getMass(*e) for e in volumes)/volume
    gmsh.option.setNumber('Mesh.MeshSizeMin',1.5);gmsh.option.setNumber('Mesh.MeshSizeMax',5.)
    gmsh.model.mesh.generate(2);tags,xyz,_=gmsh.model.mesh.getNodes();xyz=np.array(xyz).reshape(-1,3)
    mapping={int(v):i+1 for i,v in enumerate(tags)};faces=[]
    types,_,elements=gmsh.model.mesh.getElements(2)
    for kind,values in zip(types,elements):
        if kind==2:faces.extend(np.array(values).reshape(-1,3).tolist())
    target=out/(name+'.obj')
    with target.open('w') as f:
        for x in xyz*.001:f.write('v '+' '.join(map(str,x))+'\n')
        for face in faces:f.write('f '+' '.join(str(mapping[v]) for v in face)+'\n')
    report['parts'][name]={'source':str(p.relative_to(src)),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'volume_m3':volume*1e-9,'centroid_m':(centroid*.001).tolist(),'bbox_min_m':(xyz.min(0)*.001).tolist(),'bbox_max_m':(xyz.max(0)*.001).tolist(),'triangles':len(faces)}
gmsh.finalize();(out/'provenance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
