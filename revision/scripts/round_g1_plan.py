"""Select a base height for each comparator before flight and freeze the references."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from round_g1_model import DATA, D, LAUNCH, P, SHIFT, TARGET, Geometry, make_plant


def smooth(t, a, b):
    u = np.clip((t-a)/(b-a), 0., 1.)
    return u*u*u*(10. + u*(-15. + 6.*u))


def candidates(geo, theta, heights, mode):
    # One FK per time for all gate heights. A vertical base translation changes
    # only z; use the horizontal box gap to discard pairs already over 25 cm.
    init = P['initial_hover_z']
    end = 20 if mode.startswith('variable') else theta
    geo.place([0., 0., 0.], end)
    a = dict(gate_deg=theta, end_deg=end, z=init, end_xyz=(TARGET-geo.tip()[0]).tolist())
    caps = np.full((2, len(heights)), .25)
    for t in np.arange(12., 46.01, .20):
        _, xyz, angle = reference(t, a, mode)
        geo.place(xyz+LAUNCH, angle)
        lo, hi = geo.bounds()
        sep = np.maximum(np.maximum(lo[:,None,:2]-geo.ghi[:,:2], geo.glo[:,:2]-hi[:,None,:2]),0.)
        xy2 = (sep*sep).sum(axis=2)
        ids, walls = np.nonzero(xy2 < .25**2)
        if not len(ids):
            continue
        coeff = smooth(t, *P['lift_s'])-smooth(t, *P['lower_s'])
        dz = coeff*(heights-init)
        vertical = np.maximum(np.maximum((lo[ids,2]-geo.ghi[walls,2])[:,None]+dz,
                                         (geo.glo[walls,2]-hi[ids,2])[:,None]-dz),0.)
        squared = vertical**2 + xy2[ids,walls,None]
        for i, mask in enumerate((geo.cat[ids]=='body', geo.cat[ids]!='body')):
            if np.any(mask):
                caps[i] = np.minimum(caps[i], np.sqrt(np.min(squared[mask],axis=0)))
    return [dict(a,z=round(float(z),4),body=float(b),arm_tool=float(c),
                 slack=float(min(b-P['gate_body_margin_m'],c-P['gate_arm_tool_margin_m'])))
            for z,b,c in zip(heights,*caps)]


def select(rows):
    best = max(r['slack'] for r in rows)
    short = [r for r in rows if r['slack'] >= max(0.,best-.005)] if best>=0. else [r for r in rows if r['slack'] >= best-1e-9]
    return min(short, key=lambda r:(r['gate_deg'], abs(r['z']-P['initial_hover_z']), -r['slack']))


def reference(t, a, mode):
    init = P['initial_hover_z']
    if t is None:
        return 'preflight', np.array([0., 0., init])-LAUNCH, 0.
    g, end = a['gate_deg'], a['end_deg']
    # One continuous progress curve through the gate. Warp only the second half so
    # each method can place its own base at the common terminal TCP.
    r = smooth(t, 16., 42.)
    x = 2.*r + (a['end_xyz'][0]-2.)*smooth(t, 29., 42.)
    z = init+(a['z']-init)*smooth(t, *P['lift_s'])+(a['end_xyz'][2]-a['z'])*smooth(t, *P['lower_s'])
    if mode == 'fixed':
        angle = g*smooth(t, 8., 14.)
    elif mode == 'straight':
        angle = 0.
    else:
        bend, ret = P['timed_bend_return_s' if mode=='variable_timed' else 'nominal_bend_return_s']
        angle = g*smooth(t, *bend)+(end-g)*smooth(t, *ret)
    phase = ('liftoff' if t<8 else 'conditioning' if t<16 else 'approach' if t<27 else
             'traverse' if t<33 else 'depart' if t<42 else 'terminal')
    return phase, np.array([x, 0., z])-LAUNCH, float(angle)


def wide_reference():
    archive = np.load(str(D/'receiver_trajectory.npz'))
    times, states = archive['times'], archive['states']
    def fn(t):
        elapsed = 0. if t is None else max(0., t-16.)
        xyz = np.array([np.interp(elapsed, times, states[:, i]) for i in range(3)])+SHIFT
        return ('preflight' if t is None else 'approach' if t<16 else
                'wide_traverse' if elapsed<times[-1] else 'terminal'), xyz-LAUNCH, 0.
    return fn


def main():
    DATA.mkdir(exist_ok=True)
    geo = Geometry(make_plant())
    z0, z1, step = P['grid_pass_height_m']
    heights = np.arange(z0, z1+.5*step, step)
    selected, pool, sweep = {}, {}, {}
    for mode, angles in [('straight',[0]), ('fixed', P['fixed_grid_deg']), ('variable_nominal', P['pass_grid_deg'])]:
        rows = [r for ang in angles for r in candidates(geo, ang, heights, mode)]
        pool[mode] = rows
        q = select(rows).copy()
        selected[mode] = q
    selected['variable_timed'] = selected['variable_nominal'].copy()
    for mode, a in selected.items():
        low = np.full(2, np.inf)
        for t in np.arange(14., 46.001, .05):
            _, pos, angle = reference(t, a, mode)
            geo.place(pos+LAUNCH, angle)
            gap = geo.distances(*geo.bounds()).min(axis=1)
            low = np.minimum(low, [gap[geo.cat=='body'].min(),gap[geo.cat!='body'].min()])
        sweep[mode] = dict(sample_dt_s=.05, body_lower_m=low[0], arm_tool_lower_m=low[1],
                           meets_nominal_margins=bool(low[0]>=P['gate_body_margin_m'] and low[1]>=P['gate_arm_tool_margin_m']))
    record = dict(selected=selected, nominal_sweep=sweep, candidate_counts={k:len(v) for k,v in pool.items()},
                  wide_accepted_npz_sha256=hashlib.sha256((D/'receiver_trajectory.npz').read_bytes()).hexdigest(),
                  geom_counts={k:int(sum(geo.cat==k)) for k in ('body','arm','tool')})
    for name, data in [('plans.json',record),('candidate-grid.json',pool)]:
        (DATA/name).write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
