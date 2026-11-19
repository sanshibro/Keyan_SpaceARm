"""Anonymously verify every byte of the pinned, externally served meeting assets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def verify(job):
    name, item = job
    for attempt in range(4):
        try:
            request = Request(item['external_url'],headers={'User-Agent':'Research-archive/1.0'})
            with urlopen(request,timeout=55) as response:
                digest=sha256()
                count=0
                for block in iter(lambda:response.read(262144), b''):
                    digest.update(block)
                    count+=len(block)
            if count!=item['bytes'] or digest.hexdigest()!=item['sha256']:
                raise ValueError('public bytes differ for '+name)
            return name,item
        except OSError:
            if attempt==3:
                raise
            time.sleep(2**attempt)


def main(stage, raw_commit, mirror_commit):
    base='https://raw.githubusercontent.com/sanshibro/Keyan_SpaceARm/'
    if mirror_commit:
        folder=stage/'canonical/dist/archive/2026-11-19-g1-online'
        target='revision/meeting_archive/archive/2026-11-19-g1-online/'
        jobs={}
        for name,relative in [('meeting.zip','meeting.zip'),('all-traces.npz','media/all-traces.npz')]:
            p=folder/relative
            jobs[name]={'sha256':sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,
                        'external_url':base+mirror_commit+'/'+target+relative}
        output=stage/'public-mirror-verification.json'
    else:
        jobs=json.loads((stage/'all-evidence-index.json').read_bytes())
        assert len(jobs)==21
        for n,entry in jobs.items():
            entry['external_url']=base+raw_commit+'/revision/data/round_g1_online/evidence/'+n
        output=stage/'public-raw-verification.json'
    with ThreadPoolExecutor(max_workers=5) as pool:
        verified=dict(pool.map(verify,sorted(jobs.items())))
    output.write_text(json.dumps(verified,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('Anonymous GET and SHA-256 matched',len(verified),'files,',sum(f['bytes'] for f in verified.values()),'bytes')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',type=Path)
    parser.add_argument('raw_commit')
    parser.add_argument('--mirror-commit')
    args=parser.parse_args()
    main(args.stage,args.raw_commit,args.mirror_commit)
