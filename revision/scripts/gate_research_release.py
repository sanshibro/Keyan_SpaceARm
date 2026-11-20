"""Package and import the independently audited gate review without replacing earlier rounds."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from tempfile import NamedTemporaryFile

DATA=PurePosixPath('revision/data/round_gate_feedback')
ARCH=PurePosixPath('revision/meeting_archive/archive/2026-11-20-gate-feedback')
DEC=('cohort-summary.json','case-evidence-audit.json','gate-physical-audit.csv','native-log-audit.json')
FIGURES=('gate_keyframes','gate_four_metrics','gate_method_details','gate_paired_outcomes','gate_state_barcode')
REVIEW=('gate_review_manifest.json','visual_manifest.json','video-verification.json','wrist-window.json',
        'gate_overview.png',*[stem+ext for stem in FIGURES for ext in ('.png','.pdf','.svg')])


def pack(stage,archive):
    gate=stage/'canonical-release/dist/archive/2026-11-20-gate-feedback'
    entries={ARCH/p.relative_to(gate).as_posix():p for p in sorted(gate.rglob('*')) if p.is_file()}
    for n in DEC:entries[DATA/'decisions'/n]=stage/'decisions'/n
    for n in REVIEW:entries[DATA/'review'/n]=stage/'review'/n
    if len(json.loads((stage/'decisions/metrics.json').read_bytes())['cases'])!=60:
        raise ValueError('Incomplete registered cohort')
    temp=archive.with_suffix('.part')
    with tarfile.open(temp,'w:gz',compresslevel=2) as tf:
        for n,p in entries.items():
            if not p.is_file() or p.is_symlink():raise FileNotFoundError(p)
            tf.add(p,arcname=n.as_posix(),recursive=False)
    temp.replace(archive)
    return {n.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for n,p in entries.items()}


def install(repo,archive,manifest):
    expected=json.loads(manifest.read_bytes())
    seen=set()
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf:
            n=PurePosixPath(member.name)
            allowed=(ARCH in n.parents or (n.parent==DATA/'review' and n.name in REVIEW) or
                     (n.parent==DATA/'decisions' and n.name in DEC))
            if not member.isfile() or not allowed or n.is_absolute() or '..' in n.parts or member.name not in expected or member.name in seen:
                raise ValueError('Unexpected research release member: '+member.name)
            p=(repo/n).resolve()
            if repo.resolve() not in p.parents:raise ValueError('Path escapes research worktree')
            raw=tf.extractfile(member).read();digest=hashlib.sha256(raw).hexdigest()
            if expected[member.name]!=digest:raise IOError('Incoming media fingerprint: '+member.name)
            if p.exists():
                if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise FileExistsError('Existing research record differs: '+str(n))
            else:
                p.parent.mkdir(parents=True,exist_ok=True)
                with NamedTemporaryFile(dir=p.parent,delete=False) as f:
                    f.write(raw);temp=Path(f.name)
                temp.replace(p)
            seen.add(member.name)
    if seen!=set(expected):raise IOError('Research release incomplete')
    print('Verified',len(seen),'figures, audit results and meeting archive files')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('pack','install'));ap.add_argument('root',type=Path)
    ap.add_argument('archive',type=Path);ap.add_argument('manifest',type=Path)
    a=ap.parse_args()
    if a.mode=='pack':
        values=pack(a.root.resolve(),a.archive)
        a.manifest.write_text(json.dumps(values,indent=2)+'\n',encoding='utf-8')
        print(len(values),'verified local release artifacts')
    else:install(a.root.resolve(),a.archive,a.manifest)
