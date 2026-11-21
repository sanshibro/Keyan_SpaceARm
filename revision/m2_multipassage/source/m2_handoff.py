"""Original validated QuinticBendingBridge with analytic execution derivatives."""
import numpy as np
from super_handoff import QuinticBendingBridge,coordinates,rates,accelerations,review
from uav_continuum_whole_body_v2.execution import quintic_bridge_coefficients,evaluate_quintic_bridge,whole_body_switch_deltas

class SuffixCurve:
    """An unchanged suffix of an already certified absolute-time trajectory."""
    def __init__(self,source,offset):
        self.source=source;self.offset=float(offset)
        if not 0<=self.offset<source.duration:raise ValueError('Expired local candidate at receiving time')
        self.duration=source.duration-self.offset
        self.span_boundaries=np.unique(np.r_[0.,source.span_boundaries[source.span_boundaries>offset]-offset,self.duration])
    def evaluate(self,t,derivative=0):return self.source.evaluate(np.asarray(t)+self.offset,derivative)

def qualify_timed(adapter,old,raw_curve,offset,bridge_duration=.5):
    q,v,a=old;candidate=SuffixCurve(raw_curve,offset)
    direct=review(adapter,q,v,candidate)
    if direct.feasible:
        target=tuple(candidate.evaluate([0.],d)[0] for d in range(3))
        return None,direct,dict(kind='C1_direct',duration=0.,trajectory_offset_s=offset,
                  before=boundary_record(q,v,a),after=boundary_record(*target)),offset
    target_offset=offset+bridge_duration;candidate=SuffixCurve(raw_curve,target_offset)
    bridge=make_bridge(q,v,a,candidate,bridge_duration)
    proof=review(adapter,q,v,candidate,bridge,start_acceleration=a)
    target=tuple(candidate.evaluate([0.],d)[0] for d in range(3))
    return bridge,proof,dict(kind='C2_quintic',duration=bridge.duration,trajectory_offset_s=target_offset,
             before=boundary_record(q,v,a),after=boundary_record(*target),coefficients=bridge.c.tolist(),
             direct_rejection=direct.failure_reason),target_offset

def make_bridge(q,v,a,successor,duration):
    target=successor.evaluate([0.],0)[0];tv=successor.evaluate([0.],1)[0];ta=successor.evaluate([0.],2)[0]
    end=coordinates(target);start=coordinates(q);end[3]=start[3]+(end[3]-start[3]+np.pi)%(2*np.pi)-np.pi
    coeff=quintic_bridge_coefficients(start,rates(q,v),accelerations(q,v,a),end,rates(target,tv),accelerations(target,tv,ta),duration)
    return QuinticBendingBridge(coeff,duration)

def sample_bridge(bridge,t):
    x=evaluate_quintic_bridge(bridge.c,[t],0)[0];v=evaluate_quintic_bridge(bridge.c,[t],1)[0];a=evaluate_quintic_bridge(bridge.c,[t],2)[0]
    theta=np.linalg.norm(x[4:]);phi=np.arctan2(x[5],x[4]);unit=np.array([np.cos(phi),np.sin(phi)]);tangent=np.array([-unit[1],unit[0]])
    if theta<1e-9:tr=float(np.linalg.norm(v[4:]));pr=0.;ta=float(unit@a[4:]);pa=0.
    else:
        tr=float(unit@v[4:]);pr=float(tangent@v[4:]/theta)
        ta=float(unit@a[4:]+theta*pr*pr);pa=float((tangent@a[4:]-2*tr*pr)/theta)
    return np.r_[x[:4],theta,phi],np.r_[v[:4],tr,pr],np.r_[a[:4],ta,pa]

def boundary_record(q,v,a):
    return dict(q=np.asarray(q).tolist(),dq=np.asarray(v).tolist(),ddq=np.asarray(a).tolist(),
                qb=coordinates(q).tolist(),dqb=rates(q,v).tolist(),ddqb=accelerations(q,v,a).tolist())

def qualify(adapter,old,new,bridge_duration=.5):
    q,v,a=old;curve=new['curve'];target=tuple(curve.evaluate([0.],d)[0] for d in range(3))
    direct=review(adapter,q,v,curve)
    record=dict(before=boundary_record(q,v,a),after=boundary_record(*target),gaps=whole_body_switch_deltas(q,v,target[0],target[1]))
    if direct.feasible:return None,direct,dict(record,kind='C1_direct',duration=0.)
    bridge=make_bridge(q,v,a,curve,bridge_duration)
    result=review(adapter,q,v,curve,bridge,start_acceleration=a)
    return bridge,result,dict(record,kind='C2_quintic',duration=bridge.duration,coefficients=bridge.c.tolist(),
                              direct_rejection=direct.failure_reason)
