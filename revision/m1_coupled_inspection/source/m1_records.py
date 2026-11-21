"""Package registered native evidence for the NUC branch and the local meeting copy."""
import argparse,hashlib,json,tarfile
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def put(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def package(root,archive):
    metrics=json.loads((root/'matrix.json').read_text());reg=json.loads((root/'registered.json').read_text())
    assert len(metrics)==15;keys=set();summary=[]
    include={'registered.json','matrix.json','plan-registration.json'}
    for score in metrics:
        name=score['case']['case'];case=root/name;flt=case/'flight';pro=json.loads((flt/'provenance.json').read_text())
        assert name not in keys and pro['mission']['scene']==reg['scene_sha256'];keys.add(name)
        files=['plan/decision.json','flight/run.json','flight/provenance.json','flight/events.jsonl',
               'flight/control-summary.json','flight/physics.npz','flight/control.npz','flight/flight.ulg','flight/flight.bag',
               'flight/optics/observations.npz','flight/optics/index.jsonl','flight/optics/shared-scene.json','audit/metrics.json','audit/series.npz']
        files += [str(x.relative_to(case)) for x in (flt/'sensors').iterdir() if x.suffix in ('.npz','.json')]
        if pro['mission']['plan_sha256']!=sha(case/'plan/decision.json'):raise ValueError('Plan identity: '+name)
        if json.loads((case/'plan/decision.json').read_text())['accepted']:files.append('plan/certified.npz')
        files += ['plan/'+n for n in ('seed.npy','body-reference.npy') if (case/'plan'/n).is_file()]
        files += ['flight/optics/'+n for n in ('native.csv','native.log','settings.yaml','fixed-calibration.json','complete.json','manifest.json')]
        # Every native exposure is re-hashed against its producer's index. This
        # also protects the RGB-D evidence retained on the NUC for each flight.
        source=flt/'optics';idx=[json.loads(z) for z in (source/'index.jsonl').read_text().splitlines()]
        rgb=hashlib.sha256();dep=hashlib.sha256()
        for o in idx:
            rd=sha(source/o['rgb']);dd=sha(source/o['depth'])
            if rd!=o['rgb_sha256'] or ('depth_sha256' in o and dd!=o['depth_sha256']):raise ValueError(name+': exposure '+str(o['frame']))
            rgb.update(bytes.fromhex(rd));dep.update(bytes.fromhex(dd))
        origins={}
        for n in files:
            p=case/n
            if not p.is_file():raise FileNotFoundError(p)
            origins[n]={'sha256':sha(p),'bytes':p.stat().st_size};include.add(name+'/'+n)
        summary.append({'case':name,'exposures':len(idx),'rgb_ordered_sha256':rgb.hexdigest(),'depth_ordered_sha256':dep.hexdigest(),'components':origins})
    put(root/'evidence.json',{'scene':reg['scene_sha256'],'source_commit':reg['source_commit'],'native_flight_count':len(summary),'trials':summary})
    include.add('evidence.json');archive.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(str(archive),'w:gz',compresslevel=3) as tar:
        for n in sorted(include):tar.add(str(root/n),arcname=n,recursive=False)
    print('registered',len(keys),'files',len(include),'bytes',archive.stat().st_size,'sha256',sha(archive))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('folder',type=Path);a.add_argument('archive',type=Path);v=a.parse_args();package(v.folder,v.archive)
