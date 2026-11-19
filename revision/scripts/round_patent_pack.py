"""Archive the exact inputs and delayed estimator outputs used by the reviewer."""
import hashlib,tarfile
from round_patent_camera import OUT
from round_wrist_fusion import CASES,savej
from round_shape_pixels import OUTPUT
def run():
    evidence=OUT/'evidence';evidence.mkdir(exist_ok=True)
    obs={}
    for name in ('observations','camera','support'):
        for p in sorted((OUT/name).rglob('*')):
            if p.is_file() and p.suffix in ('.npz','.csv','.json'):obs[p.relative_to(OUT).as_posix()]=p
    for key in ['visual-calibration','strain-variance','variance-bench']:obs[key+'.json']=OUT/(key+'.json')
    res={}
    for p in sorted((OUT/'traces').rglob('*.npz')):res[p.relative_to(OUT).as_posix()]=p
    for case in CASES:res['scoring-only/'+case+'/scoring_truth.npz']=OUTPUT/'runs'/case/'scoring_truth.npz'
    for key in ['paired-scores','case-scores','support-scores','teacher-and-recovery','audit']:res[key+'.json']=OUT/(key+'.json')
    index={}
    for name,members in [('confidence-inputs.tar.gz',obs),('confidence-results.tar.gz',res)]:
        dst=evidence/name
        with tarfile.open(dst,'w:gz',compresslevel=5) as tf:
            for arc,p in members.items():tf.add(p,arcname=arc,recursive=False)
        with tarfile.open(dst,'r:gz') as tf:
            assert len(tf.getmembers())==len(members)
            for f in tf.getmembers():
                if hashlib.sha256(tf.extractfile(f).read()).digest()!=hashlib.sha256(members[f.name].read_bytes()).digest():raise ValueError(f.name)
        index[name]=dict(bytes=dst.stat().st_size,members=len(members),sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
    savej(OUT/'evidence-index.json',index)
    print(index)
if __name__=='__main__':run()
