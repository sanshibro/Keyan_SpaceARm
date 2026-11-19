"""Acquire three fresh loopback PX4 SITL / MuJoCo runs using the registered routes."""
import fcntl,json,tempfile
from pathlib import Path
import numpy as np
from round_g1_online_run import Route,Policy,PLAN,OLD,make_plant,phase_time
from round_f_sitl import run,digest
from round_g1_model import ROOT
PROTO=ROOT/'revision/protocols/round_health_protocol.json'
SPEC=json.loads(PROTO.read_bytes())
OUT=ROOT/'revision/data/round_strain_health'
class TestRoute(Route):
    def __init__(self,case):
        super().__init__(case['route'])
        self.lateral=case['lateral_m'];self.scale=case['bend_scale']
        self.path[:,1]+=self.side(self.grid)
    def side(self,t):
        u=np.clip((np.asarray(t)-14.)/30.,0,1)
        return self.lateral*np.sin(np.pi*u)**2
    def base(self,t):
        phase,ref,angle=super().base(t)
        if t is None:return phase,ref,angle
        ref=ref.copy();ref[1]+=self.side(phase_time(t,self.condition))
        return phase,ref,angle*self.scale
def main():
    entries=SPEC['flight'];folder=OUT/'runs';folder.mkdir(parents=True,exist_ok=True)
    with (Path(tempfile.gettempdir())/'super-qingyu-sitl-instance-6.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for entry in entries:
            name=entry['case'];dest=folder/name
            if dest.exists():
                if (dest/'run.json').exists() and json.loads((dest/'run.json').read_bytes())['error'] is None and (dest/'flight.ulg').exists():continue
                raise RuntimeError('Inspect incomplete run before retry: '+str(dest))
            route=TestRoute(entry);policy=Policy(dest,route,'phase')
            try:
                print('Starting native flight',name,flush=True)
                run(dest,name,.4,reference_fn=route.base,score_end=49.,noise_seed=entry['seed'],protocol_name=PROTO.name,
                    mission_record=dict(**entry,plan_sha256=digest(OLD/'plans.json'),observer='Held out for estimation; shape reference only.'),
                    plant_factory=make_plant,shape_policy=policy)
            finally:policy.close()
            print('Recorded',name,(dest/'flight.ulg').stat().st_size,'ULog bytes',flush=True)
if __name__=='__main__':main()
