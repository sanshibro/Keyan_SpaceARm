"""Pack only the original per-flight evidence, preserving the uncompressed SHA-256."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('round',type=Path)
root=parser.parse_args().round
audit=json.loads((root/'audit.json').read_text())
dest=root/'evidence'
dest.mkdir(exist_ok=True)
manifest={}
for case, record in audit['runs'].items():
    raw=root/'runs'/case
    files=sorted(list(record['sha256'])+['px4-console.txt'])
    pack=dest/(case+'.tar.gz')
    if not pack.exists():
        temp=pack.with_suffix('.partial')
        with tarfile.open(str(temp),'w:gz',compresslevel=5) as tar:
            for n in files:
                tar.add(raw/n,arcname=case+'/'+n,recursive=False)
        temp.replace(pack)
    with tarfile.open(str(pack),'r:gz') as tar:
        for n, expected in record['sha256'].items():
            assert hashlib.sha256(tar.extractfile(case+'/'+n).read()).hexdigest()==expected
    manifest[pack.name]={'bytes':pack.stat().st_size,'sha256':hashlib.sha256(pack.read_bytes()).hexdigest()}
(root/'evidence-index.json').write_text(json.dumps(manifest,indent=2)+'\n')
