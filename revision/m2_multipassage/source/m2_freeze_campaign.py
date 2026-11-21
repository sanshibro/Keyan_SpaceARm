"""Snapshot the complete M2 runtime for a reproducible development/formal campaign."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

STAGE=Path(__file__).resolve().parents[1]

def register(tag,records):
    if not tag or Path(tag).name!=tag:raise ValueError('One attempt name is required')
    parent=records.resolve()/'runtime-snapshots'/tag
    parent.parent.mkdir(parents=True,exist_ok=True)
    parent.mkdir()
    (parent/'source').mkdir();(parent/'qa/static').mkdir(parents=True)
    for f in sorted((STAGE/'source').iterdir()):
        if f.suffix in ('.py','.cc','.json'):
            shutil.copy2(str(f),str(parent/'source'/f.name))
    for name in ('qa/registered-collision-scene.json','qa/static/certified.json','qa/static/native-crosscheck.json'):
        shutil.copy2(str(STAGE/name),str(parent/name))
    scene=Path(os.environ.get('M2_SCENE',STAGE/'scene')).resolve()
    doc={'scene':str(scene),'scene_manifest_sha256':hashlib.sha256((scene/'manifest.json').read_bytes()).hexdigest(),
         'source':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (parent/'source').iterdir()}}
    (parent/'snapshot.json').write_text(json.dumps(doc,indent=2)+'\n')
    return parent,scene

def main():
    p=argparse.ArgumentParser();p.add_argument('tag');p.add_argument('records',type=Path)
    p.add_argument('--development',action='store_true');p.add_argument('--qualification',type=Path);p.add_argument('--instance',type=int,default=4)
    a=p.parse_args()
    if a.development==bool(a.qualification):p.error('Specify exactly one campaign mode')
    if a.development:
        root,scene=register(a.tag,a.records);gate=root/'qa/static/certified.json'
    else:
        gate=a.qualification.resolve()
        if not gate.is_file():raise FileNotFoundError(gate)
        devtag=gate.parent.name
        root=a.records.resolve()/'runtime-snapshots'/devtag
        doc=json.loads((root/'snapshot.json').read_text());scene=Path(doc['scene'])
        for name,expected in doc['source'].items():
            if hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()!=expected:raise ValueError('Modified snapshot '+name)
    os.environ['M2_SCENE']=str(scene)
    os.environ['PYTHONPATH']=str(root/'source')+os.pathsep+os.environ.get('PYTHONPATH','')
    mode='development' if a.development else 'formal'
    cmd=[sys.executable,'-u',str(root/'source/m2_campaign.py'),mode,a.tag,str(a.records.resolve()),str(gate),'--instance',str(a.instance)]
    print('Staged',mode,a.tag,root,flush=True)
    os.execv(sys.executable,cmd)

if __name__=='__main__':main()
