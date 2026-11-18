"""Recover the historical method's denominators and latency from the original per-attempt files."""
import argparse, csv, hashlib, json
from pathlib import Path
from collections import Counter
import numpy as np

EXP=Path('xtdrone_plan/experiments/method_comparison/v21_e3_final_450_commite62b6f2')
def inspect(original,repo):
    timings=[];failures=[];evidence=[];tasks=set();family={}
    for file in sorted((original/EXP).glob('*/closed_loop_runs/*__rolling_wb_v2__*/acceptance_report.json')):
        task=file.parent.name.split('__')[0]
        assert task not in tasks
        tasks.add(task);doc=json.loads(file.read_text());assert doc['method']=='rolling_wb_v2'
        fam=task.rsplit('_',2)[0]
        cnt=family.setdefault(fam,[0,0]);cnt[0]+=int(doc['passed']);cnt[1]+=1
        if not doc['passed']:failures.append({'scenario':task,'reason':doc['failures']})
        vals=doc['all_planning_attempt_time_s']['values'];assert len(vals)==doc['all_planning_attempt_time_s']['count']
        timings.extend(vals)
        evidence.append(dict(scenario=task,count=len(vals),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    handoff=repo/EXP/'paper_artifacts/source_data/handoff_continuity_samples.csv'
    modes=Counter(r['handoff_mode'] for r in csv.DictReader(handoff.open()))
    assert len(tasks)==90 and sum(x[0] for x in family.values())==89 and len(timings)==494 and modes=={'C1':339,'C2':52}
    vals=np.asarray(timings);assert np.isfinite(vals).all()
    return dict(software='historical PX4/Gazebo, second-order 0.22 m arm response',family_completed_and_attempted=family,
        failures=failures,all_attempts=len(vals),median_s=float(np.median(vals)),p95_s=float(np.percentile(vals,95)),max_s=float(max(vals)),
        over_15s=int(np.sum(vals>15)),handoffs=dict(modes),handoff_sha256=hashlib.sha256(handoff.read_bytes()).hexdigest(),inputs=evidence)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('original',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=inspect(a.original,Path(__file__).resolve().parents[2]);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ('inputs','failures')})
