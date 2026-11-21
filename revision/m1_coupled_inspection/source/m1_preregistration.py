"""Register immutable planning inputs and align paired flights to one common controller."""
import hashlib,json
from pathlib import Path
from m1_layout import METHODS,SEEDS
from m1_plan import writej
def apply(earlier,later):
    a,b=Path(earlier),Path(later)
    items={}
    for seed in SEEDS:
        for method in METHODS:
            name='%s-%d'%(method,seed);p=a/name/'plan';d=b/name/'plan'
            f=[q for q in p.iterdir() if q.is_file()]
            hashes={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in f}
            assert all(hashlib.sha256((d/n).read_bytes()).hexdigest()==v for n,v in hashes.items())
            items[name]=hashes
    writej(b/'plan-registration.json',dict(plan_origin_commit=json.loads((a/'registered.json').read_text())['source_commit'],
        change_basis='Only development seed 26112107 was scored for controller integration. The two prior pre-qualification flight logs are preserved separately.',
        controller='One feedback and clearance-acceleration law for all methods; true flight measurements scored independently.',
        baselines='AET-Seq. is the in-house body-first/then-shape restriction of this planner, inspired by aerial manipulation planning; authors release and implementation: https://github.com/arclab-hku/AET',
        plans=items))
if __name__=='__main__':
    import sys
    apply(sys.argv[1],sys.argv[2])
