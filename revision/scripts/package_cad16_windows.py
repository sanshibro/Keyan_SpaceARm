"""Portable read-only native meshes; no SolidWorks or original-cache mutation."""
from pathlib import Path,PureWindowsPath
import json,hashlib,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent
CAD=ROOT.parent/'16_机腹模块化拆装/16_机腹模块化拆装/配色与爆炸渲染_20261116'
origin=CAD/'render/input/gauge60_20261117/manifest_gauge60.json'
j=json.loads(origin.read_text('utf-8-sig'));rows=[]
out=ROOT/'cad16_native_meshes.zip'
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for i,p in enumerate(j['components']):
        mesh=CAD/'render/cache'/f'{i:04}_000.ply'
        entry={k:p[k] for k in ['id','name','group','material_role','color','metallic','roughness','opacity','actual_world_bounds_m'] if k in p}
        entry.update(index=i,file=f'meshes/{mesh.name}',source_part=PureWindowsPath(p.get('source','')).name,sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),render_only=p.get('render_only',False))
        z.write(mesh,entry['file'])
        for tag in ['_texture.png','_texture_alpha.png']:
            t=mesh.parent/(f'{i:04}_000'+tag)
            if t.exists():z.write(t,'meshes/'+t.name)
        rows.append(entry)
    # Body includes all fixed components, notably landing gear and lidar/computer.
    fixed=[p for p in rows if p['group']!='arm' and 'actual_world_bounds_m' in p]
    lo=np.min([p['actual_world_bounds_m'][0] for p in fixed],axis=0);hi=np.max([p['actual_world_bounds_m'][1] for p in fixed],axis=0)
    receipt=dict(schema_version=1,source='native SolidWorks 16-series tessellation used for approved hardware figures',source_manifest_sha256=hashlib.sha256(origin.read_bytes()).hexdigest(),units='m',up_axis='Y',front_axis='+Z',configuration_bend_deg=90.,geometric_scale=1.,original_geometry_modified=False,dynamics_calibrated=False,rigid_bounds_m=[lo.tolist(),hi.tolist()],rigid_extent_m=(hi-lo).tolist(),physics_identity='visual-only asset package; collision/inertia/actuation require validation',components=rows)
    z.writestr('manifest.json',json.dumps(receipt,ensure_ascii=False,indent=2))
(ROOT/'cad16_geometry_manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(zip_bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),parts=len(rows),rigid_extent_m=receipt['rigid_extent_m'])))
