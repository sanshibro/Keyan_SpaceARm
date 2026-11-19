"""Independent causal, uncertainty, plane, recovery and presentation-clock checks."""
import numpy as np
from patent_confidence import angles,b_from,wrap,fuse
from round_patent_eval import GAUGES,packet,optical
from round_patent_camera import OUT
from round_wrist_fusion import CASES,readj,savej

def synthetic(p):
    t=np.arange(160)*.02;cap=t[::2];theta=np.full(len(t),42.)
    b=b_from(np.column_stack((theta,np.full(len(t),179.))))
    v=b_from(np.column_stack((theta[::2],np.full(len(cap),-179.))))
    vq=np.ones(len(cap));fresh=np.ones(len(t),bool);s2=np.ones((len(t),2))*4
    def apply(tstate,ss,ff,vv,cv):
        return fuse(t,tstate,ff,ss,np.zeros(len(t)),cap,vv,cv,np.zeros(len(cap),bool),p)
    z=apply(b,s2,fresh,v,vq);gap=abs(float(wrap(z['angle'][-1,1]-180)))
    assert gap<1.2
    near=b_from(np.column_stack((np.ones(len(t)),np.linspace(-170,170,len(t)))))
    straight=apply(near,s2,fresh,near[::2],vq);assert not straight['plane'].any()
    good=np.array([i<45 for i in range(len(t))]);none=apply(b,s2,good,v,np.where(cap<.6,1.,0.))
    assert (np.diff(none['posterior_sigma'][65:,0])>0).all() and not none['observed'][65:].any()
    low=apply(b,s2,fresh,v,.15*vq);bad=apply(b,s2*9,fresh,v,vq)
    weights=[float(np.nanmedian(x['joint_vis_share'][:,0])) for x in (low,z,bad)]
    assert weights[0]<weights[1]<weights[2]
    empty=apply(np.full_like(b,np.nan),s2,fresh,v,0.*vq)
    assert not empty['observed'].any() and np.isfinite(empty['angle']).all()
    return dict(wrap_error_around_180_deg=gap,near_straight_plane_updates=int(straight['plane'].sum()),
        both_absent_sigma_start_end_deg=none['posterior_sigma'][[65,-1],0].tolist(),
        simultaneous_vision_share_low_cv_normal_high_strain_variance=weights)
def run():
    p=readj(OUT/'visual-calibration.json');results={}; prefix_error=0.;update=max_theta=0;negative=count=0;probe=[]
    for case in CASES:
        v=dict(np.load(OUT/'observations'/case/'occluded.npz'))
        probe.append((case,float(v['capture'][0])))
        for g in GAUGES:
            f=dict(np.load(OUT/'observations'/case/(g+'.npz')));t=f['t']
            z=dict(np.load(OUT/'traces'/case/(g+'-occluded.npz')))
            blind=(t>=31.2-1e-8)&(t<35.2-1e-8);ix=np.flatnonzero(blind)
            assert not z['accepted'][blind].any() and not np.isfinite(z['joint'][blind]).any()
            visible=np.isfinite(z['b'][1,blind]).all(1).sum()
            # The first blind exposure invalidates pending updates. Compare all subsequent policy counters.
            diff=np.ptp(z['changes'][np.r_[ix[0]-1,ix]],axis=0)
            assert not diff.any() and visible==0
            update+=int(diff.sum());max_theta=max(max_theta,float(angles(z['b'][4])[:,0].max()))
            results[case+'/'+g]=dict(finite_outputs=int(np.isfinite(z['b'][4,blind]).all(1).sum()),ticks=int(blind.sum()),visible=int(visible))
            if g=='zero_offset':
                count+=int(blind.sum())
                for end in (28.06,33.14,35.56):
                    stop=t<=end; cam=v['capture']<=t[stop][-1]+1e-7
                    # Remove every future control tick and every future exposure, including its value.
                    short=fuse(*[a[stop] for a in packet(f)],*[a[cam] for a in optical(v)],p,mode='teacher')
                    err=np.max(np.abs(short['b']-z['b'][4,stop]));prefix_error=max(float(err),prefix_error)
                    assert err<1e-10 and np.array_equal(short['correction_log'],z['changes'][stop])
        for opt in ('natural','graded','occluded'):
            det=readj(OUT/'camera'/case/(opt+'-registration.json'))
            negative+=det['negative_geometry_acceptances']
    assert count==800 and negative==0 and max_theta<=120.00001
    savej(OUT/'audit.json',dict(synthetic=synthetic(p),blind_by_case=results,clock_prefix_max_difference_rad=prefix_error,
        checked_prefix_count=12,updates_learned_during_blackout=update,false_map_links_on_permuted_pixels=negative,
        zero_offset_blind_ticks=count,maximum_reported_bend_deg=max_theta,
        scorer='Only independent motion capture in score routines. Online estimator has no truth, command, scenario labels or dropout schedule.'))
    print('Blackout ticks',count,'12 prefix maximum',prefix_error,'peak',max_theta,flush=True)
if __name__=='__main__':run()
