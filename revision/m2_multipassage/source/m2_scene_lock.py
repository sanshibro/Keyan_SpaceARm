"""Compare the rendered assembly with the registered collision scene, by named solid."""
import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

def physical(xml):
    world=ET.parse(xml).getroot().find('worldbody')
    solids={}
    for g in world.iter('geom'):
        name=g.get('name','')
        if name.startswith('COL_'):
            if name in solids:raise ValueError('duplicate physical solid '+name)
            solids[name]=dict(sorted(g.attrib.items()))
        elif int(g.get('contype',1)) or int(g.get('conaffinity',1)):
            raise ValueError('unexpected collision geometry '+name)
    return solids

def make_record(folder):
    d=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    return {'solids':physical(folder/'cell.xml'),'gates':d['gates'],'face':d['face'],'axis':d['axis']}

def verify(folder,baseline):
    reference=json.loads(baseline.read_text(encoding='utf8'));current=make_record(folder)
    for k in ('gates','face','axis'):
        if current[k]!=reference[k]:raise ValueError('changed '+k)
    old=reference['solids'];new=current['solids']
    changed=[k for k in sorted(set(old)|set(new)) if old.get(k)!=new.get(k)]
    if changed:raise ValueError('collision geometry changed: '+', '.join(changed))
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    for name,digest in manifest['files'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise ValueError('changed asset '+name)
    if hashlib.sha256(json.dumps(manifest['files'],sort_keys=True).encode()).hexdigest()!=manifest['scene_sha256']:
        raise ValueError('scene digest mismatch')
    return {'identical_registered_collision_geoms':len(new),'physical_baseline_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),
            'gate_dimensions_identical':True,'target_pose_identical':True,'scene_sha256':manifest['scene_sha256']}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('scene',type=Path);parser.add_argument('baseline',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=verify(args.scene,args.baseline)
    message=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(message,encoding='utf8')
    print(message)
