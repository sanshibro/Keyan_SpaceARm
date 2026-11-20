"""Certify unresolved box distances on recorded poses with the full Minkowski hull.

The registered scorer remains the first evaluator on every pair. This offline
continuation runs only when that evaluator has reported an unresolved distance.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from scipy.spatial import ConvexHull
from gate_distance_audit import DistanceAudit, closest_simplex

SIGNS = np.array(list(itertools.product((-1.,1.),repeat=3)))


def boxes(c1,r1,h1,c2,r2,h2,tolerance):
    # A box difference is a zonotope with at most 64 vertex candidates. Center
    # coordinates before constructing the hull so translation cannot impair the bounds.
    origin=(c1+c2)*.5
    va=c1-origin+(SIGNS*h1)@r1.T
    vb=c2-origin+(SIGNS*h2)@r2.T
    a=np.repeat(va,8,axis=0)
    b=np.tile(vb,(8,1))
    w=a-b
    hull=ConvexHull(w)
    axes=np.stack([*r1.T,*r2.T,*[np.cross(x,y) for x in r1.T for y in r2.T]])
    axes=axes[np.linalg.norm(axes,axis=1)>1e-10]
    axes/=np.linalg.norm(axes,axis=1)[:,None]
    sat=float(np.max(np.abs(axes@(c1-c2))-np.abs(axes@r1)@h1-np.abs(axes@r2)@h2))
    max_offset=float(np.max(hull.equations[:,3]))
    if abs(sat-max_offset)>2e-8:
        raise ArithmeticError('Independent box separating axes and Minkowski support planes disagree')
    if sat>0:
        candidates=hull.simplices[hull.equations[:,3]>=-1e-10]
    else:
        candidates=hull.simplices[hull.equations[:,3]>=max_offset-1e-9]
    best=None
    for face in candidates:
        _,_,pa,pb=closest_simplex(a[face],b[face])
        gap=float(np.linalg.norm(pa-pb))
        if best is None or gap<best[0]:best=(gap,pa,pb)
    gap,pa,pb=best
    if sat>0:
        n=(pa-pb)/gap
        normal_bound=float(n@(c1-c2)-np.abs(n@r1)@h1-np.abs(n@r2)@h2)
        lo=max(0.,sat,normal_bound)-2e-10
        hi=gap+2e-10
    else:
        # Minimum translation depth is the nearest boundary of a convex set
        # containing the origin; analytic SAT and feasible facet witnesses agree.
        if abs(-sat-gap)>2.e-8:
            raise ArithmeticError('Box penetration depth and witness norm differ')
        lo=min(sat,-gap)-2e-10;hi=max(sat,-gap)+2e-10
    if lo>hi or hi-lo>tolerance:
        raise ArithmeticError('Box closest-point certificate exceeds registered numerical precision')
    return dict(lower=lo,upper=hi,distance=.5*(lo+hi),a=pa+origin,b=pb+origin,converged=True,
                iterations=len(candidates),status='exact OBB hull / separating-axis certificate')


class CertifiedDistance(DistanceAudit):
    records=[]
    def pair(self,a,b,lower=0.):
        try:
            return super().pair(a,b,lower)
        except ArithmeticError as err:
            if not str(err).startswith('Unresolved convex distance'):raise
            if any(self.m.geom_type[k]!=self.mj.mjtGeom.mjGEOM_BOX for k in (a,b)):raise
            p=self.d.geom_xpos;R=self.d.geom_xmat.reshape(-1,3,3);h=self.m.geom_size
            v=boxes(p[a],R[a],h[a],p[b],R[b],h[b],self.tolerance)
            self.records.append(dict(pose_index=self.calls,geom_a=self.names[a],geom_b=self.names[b],
                         original=str(err),lower_m=v['lower'],upper_m=v['upper'],
                         witness_a_m=v['a'].tolist(),witness_b_m=v['b'].tolist(),
                         requested_precision_m=self.tolerance,
                         qpos_sha256=hashlib.sha256(self.d.qpos.tobytes()).hexdigest()))
            v.update(geom_a=int(a),geom_b=int(b),name_a=self.names[a],name_b=self.names[b])
            self.warm[(int(a),int(b))]=v['a']-v['b']
            return v


def confirm_analytics():
    eye=np.eye(3);z=np.zeros(3);h=np.array([.5,.4,.3])
    for b,expected in [(np.array([1.0028,0,0]),.0028),(np.array([.985,0,0]),-.015),
                       (np.array([1.03,.84,.6]),.05),(np.array([1.,0,0]),0.)]:
        res=boxes(z,eye,h,b,eye,h,1e-7)
        if not res['lower']<=expected<=res['upper']:raise ValueError('Analytic box regression')
    # The same edges and distances must survive a common rotation and translation.
    from scipy.spatial.transform import Rotation
    rot=Rotation.from_rotvec([.17,-.48,.83]).as_matrix();shift=np.array([2.,-1.,4.])
    b=np.array([1.03,.84,.6]);res=boxes(shift,rot,h,shift+rot@b,rot,h,1e-7)
    if not res['lower']<=.05<=res['upper']:raise ValueError('Rotated box regression')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import fcntl
    import gate_feedback_scores as scorer
    from gate_feedback_campaign import design,OUT
    from gate_feedback_spec import dump
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pending',action='store_true');ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    confirm_analytics()
    if args.self_test:
        print('Analytic separated, overlapping, tangent and rotated OBBs verified');return
    if not args.pending:raise ValueError('An interrupted offline score is required')
    with (OUT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=design()
        progress=json.loads((OUT/'campaign-progress.json').read_bytes())
        c=progress['active'];folder=OUT/'formal'/c
        if c not in {x['case'] for x in protocol['cases']}:raise ValueError('Unregistered flight')
        if (folder/'metrics.json').is_file():raise ValueError('Existing scores remain sealed: '+c)
        if 'ArithmeticError: Unresolved convex distance' not in (OUT/'logs'/(c+'.log')).read_text(errors='replace').splitlines()[-1]:
            raise ValueError('Failure is not a convex closest-point convergence issue')
        if json.loads((folder/'run.json').read_bytes())['error'] is not None:raise ValueError('Acquisition did not finish')
        source=sha(Path(__file__))
        inputs={n:sha(folder/n) for n in ('physics.npz','flight.ulg','flight.bag','control.npz','run.json')}
        scorer.DistanceAudit=CertifiedDistance
        with TemporaryDirectory(dir=OUT,prefix='witness-') as working:
            stage=Path(working)/c
            stage.mkdir()
            for name in ('run.json','control-summary.json','physics.npz','control.npz','provenance.json',
                         'model.xml','protocol.json','flight.ulg','flight.bag','optics','sensors'):
                (stage/name).symlink_to(folder/name)
            scored=scorer.score(stage)
            receipt=dict(case=c,design_commit=protocol['design_commit'],additional_solver_sha256=source,
                         primary_scorer_sha256=scored['scorer_sha256'],inputs=inputs,
                         solutions=CertifiedDistance.records)
            if not receipt['solutions']:raise RuntimeError('The original recorded failure was not reproduced')
            if any(sha(folder/n)!=h for n,h in inputs.items()):raise IOError('Recorded physical inputs changed')
            where=OUT/'distance-certificates'/(c+'.json')
            where.parent.mkdir(exist_ok=True)
            dump(where,receipt)
            scored['distance_certificate']=dict(file='distance-certificates/'+where.name,sha256=sha(where),
                                  solver_sha256=source,certified_queries=len(receipt['solutions']))
            dump(stage/'metrics.json',scored)
            (stage/'score-trace.npz').replace(folder/'score-trace.npz')
            (stage/'metrics.json').replace(folder/'metrics.json')
        print('Certified offline:',c,len(receipt['solutions']),'box pairs; original integrator and flight logs retained')


if __name__=='__main__':
    main()
