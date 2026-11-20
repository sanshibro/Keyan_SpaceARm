"""Immutable flight protocol and provenance helpers used by acquisition and audit."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SPECFILE=ROOT/'revision/protocols/round_gate_feedback.json'
P=json.loads(SPECFILE.read_bytes())
OUT=ROOT/'revision/data/round_gate_feedback'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)

def verify_frozen():
    path=ROOT/'revision/protocols/gate_feedback_perception_lock.json'
    lock=json.loads(path.read_bytes())
    for name,h in lock['files'].items():
        if digest(ROOT/name)!=h:raise RuntimeError('Frozen input differs: '+name)
    return digest(path)
