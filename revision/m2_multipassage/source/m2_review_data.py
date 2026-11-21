"""Read-only offline M2 evidence loader. No controller or scoring side effects."""
import hashlib,json
from pathlib import Path
import numpy as np

METHODS={'straight':'Straight-Arm','fixed':'Fixed-Morphology','sequential':'Sequential',
 'unregularized':'Unregularized-6D','coupled':'Coupled Rolling','stop':'Stop-and-Go','hard':'Hard-Switch'}
COLORS={'straight':'#777777','fixed':'#9b8054','sequential':'#9b6889','unregularized':'#c17b4b','coupled':'#276981','stop':'#997337','hard':'#914e57'}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
def arrayfile(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}

class FlightRecord:
    def __init__(self,flight,audit=None,label=None):
        self.root=Path(flight);self.audit=Path(audit) if audit else None
        self.run=read(self.root/'run.json');self.provenance=read(self.root/'provenance.json');self.control_summary=read(self.root/'control-summary.json')
        self.physics=arrayfile(self.root/'physics.npz');self.control=arrayfile(self.root/'control.npz')
        self.release_us=int(self.run['release_us']);self.t=(self.physics['us']-self.release_us)*1e-6
        if len(self.t)<2 or np.any(np.diff(self.t)<=0):raise ValueError('Physics timestamps must increase')
        self.frames=rows(self.root/'optics/index.jsonl');self.frame_t=np.asarray([r['t'] for r in self.frames])
        if len(self.frames) and np.any(np.diff(self.frame_t)<=0):raise ValueError('Exposure timestamps must increase')
        mission=self.provenance['mission'];self.scene=mission['scene'];self.method=mission['case']['strategy'];self.seed=mission['case']['seed']
        self.label=label or METHODS.get(self.method,self.method);self.color=COLORS.get(self.method,'#276981')
        self.metrics=read(self.audit/'metrics.json') if self.audit and (self.audit/'metrics.json').is_file() else None
        self.series=arrayfile(self.audit/'series.npz') if self.audit and (self.audit/'series.npz').is_file() else None
        self.verified=set()
    def physics_index(self,t,exact=False):
        tick=self.release_us+int(round(float(t)*1e6));i=int(np.searchsorted(self.physics['us'],tick))
        candidates=[k for k in (i-1,i) if 0<=k<len(self.t)];i=min(candidates,key=lambda k:abs(int(self.physics['us'][k])-tick))
        if exact and abs(int(self.physics['us'][i])-tick)>1:raise ValueError('No exact physics row for exposure '+str(t))
        return i
    def exposure(self,t,verify=True,require_tolerance=.026):
        if not len(self.frames):return None
        k=int(np.argmin(abs(self.frame_t-float(t))))
        if abs(self.frame_t[k]-t)>require_tolerance:return None
        row=self.frames[k];i=self.physics_index(row['t'],exact=True)
        if verify and k not in self.verified:
            state_hash=hashlib.sha256(np.asarray(self.physics['qpos'][i],dtype=np.float64).tobytes()).hexdigest()
            if row.get('sensor_qpos_sha256')!=state_hash:raise ValueError('Wrist pixels / physical pose hash mismatch at frame '+str(row['frame']))
            for kind in ('rgb','depth'):
                path=self.root/'optics'/row[kind]
                if sha(path)!=row[kind+'_sha256']:raise ValueError('Modified native '+kind+' at frame '+str(row['frame']))
            if abs(float(row['arrival_t'])-float(row['t'])-.20)>1e-7:raise ValueError('Unexpected optical transmission delay')
            self.verified.add(k)
        return dict(row,physics_index=i,rgb_path=str(self.root/'optics'/row['rgb']),depth_path=str(self.root/'optics'/row['depth']))
    def verify_all_exposures(self):
        for stamp in self.frame_t:self.exposure(float(stamp),True,1e-8)
        return dict(frames=len(self.frames),pose_hashes_verified=len(self.verified),pixel_hashes_verified=2*len(self.verified),
                    release_us=self.release_us,scene_sha256=self.scene,exposure_to_arrival_s=.20)
    def events(self):
        return rows(self.root/'rolling/events.jsonl') if (self.root/'rolling/events.jsonl').exists() else []
    def status(self):
        if self.metrics is None:return 'unscored'
        return 'pass' if self.metrics.get('success') is True else 'fail'

def load_manifest(path):
    path=Path(path);doc=read(path);out=[]
    for item in doc.get('cases',[]):
        f=Path(item['flight']);a=Path(item['audit']) if item.get('audit') else None
        if not f.is_absolute():f=path.parent/f
        if a is not None and not a.is_absolute():a=path.parent/a
        out.append(FlightRecord(f,a,item.get('label')))
    return out
