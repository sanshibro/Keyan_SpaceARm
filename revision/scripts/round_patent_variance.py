"""Calibrate per-event angular uncertainty without opening evaluation flights."""
import numpy as np
from scipy.optimize import minimize
from round_slim_run import acquire,features_at,interp_b
from round_slim_box import OUT as SLIM
from round_shape_sensors import fit_model,predict,vector_angles,wrap_deg,stable_seed
from round_wrist_fusion import savej,readj
from round_patent_camera import OUT
def observable(theta):
    return np.sin(np.deg2rad(np.clip(theta,0,90)))**2
def predictors(adc,tare,t,point_model):
    adc=np.asarray(adc,float);x=(adc-np.asarray(tare))/250.
    bend=predict(point_model,features_at(adc,tare));angle,phi=vector_angles(bend)
    axes=np.column_stack((x[:,0]-x[:,2],x[:,1]-x[:,3]))
    common=np.column_stack((x[:,0]+x[:,2],x[:,1]+x[:,3]))
    smooth=np.zeros_like(axes);d=0*axes;memory=0*axes
    smooth[0]=axes[0]
    for i,dt in enumerate(np.diff(t),1):
        sm=-np.expm1(-dt/.24);smooth[i]=smooth[i-1]+sm*(axes[i]-smooth[i-1]);d[i]=(axes[i]-smooth[i])/.24
        g=-np.expm1(-dt/1.5);memory[i]=memory[i-1]+g*(np.abs(d[i])-memory[i-1])
    r=np.linalg.norm(axes,axis=1);amin=np.min(adc,axis=1)
    features=np.column_stack((r,r*r,axes,axes**2,common,np.abs(common),common**2,
             d,np.abs(d),memory,np.maximum(45.-amin,0.)/45.,(amin<=1).astype(float)))
    return bend,features
def scenarios(key):
    mem=dict(half_width=.02,tau_s=2.,play_weight=.5,transient_weight=.02)
    rng=np.random.default_rng(stable_seed('patent-cal-v1-'+key));p=.3
    for kind in ('nominal','loop','offset','mixed'):
        t,truth,d=acquire(key,p,memory=mem if kind in ('loop','mixed') else None)
        adc=d['adc'].astype(float)
        if kind in ('offset','mixed'):
            # Training offsets are drawn per episode, with the measured old box return as scale.
            profile=np.asarray(readj(SLIM/'box-profile.json')['offset_stress_counts'])
            offset=profile*rng.uniform(.35,1.30,4)
            if kind=='mixed':offset=offset*rng.choice([-1,1],4)
            onset=rng.uniform(.12,.36)*(t[-1]-t[0])+t[0]
            a=np.clip((d['t']-onset)/rng.uniform(.65,2.7),0,1)
            adc=np.rint(np.clip(adc+a[:,None]*offset,0,4095))
        yield kind,adc,d['tare'],d['t'],interp_b(d['t'],t,truth)
def design(X,mean,scale):
    return np.column_stack((np.ones(len(X)),np.clip((X-mean)/scale,-7,7)))
def head(X,y):
    # Linear log-variance head fitted by heteroscedastic Gaussian likelihood.
    y2=np.asarray(y)**2
    def objective(w):
        raw=X@w;ell=np.clip(raw,-6,9);ratio=y2*np.exp(-ell)
        loss=.5*np.mean(ell+ratio)+.008*np.sum(w[1:]**2)
        grad=.5*X.T@((1-ratio)*((raw>=-6)&(raw<=9)))/len(y)+.016*np.r_[0,w[1:]]
        return loss,grad
    start=np.zeros(X.shape[1]);start[0]=np.log(max(np.mean(y2),.04))
    fit=minimize(objective,start,jac=True,method='L-BFGS-B',options={'maxiter':400,'ftol':1e-9})
    if not fit.success:raise RuntimeError(fit.message)
    return fit.x
def var_eval(model,X):
    mat=design(X,np.asarray(model['center']),np.asarray(model['scale']))
    return np.clip(np.exp(np.clip(mat@np.asarray(model['coef']).T,-6,9))*np.asarray(model['validation_scale']),[.12**2,.3**2],[45.**2,150.**2])
def errors(b,y):
    a,p=vector_angles(b);ref,q=vector_angles(y)
    return np.column_stack((a-ref,wrap_deg(p-q))),ref>=5
def main():
    OUT.mkdir(exist_ok=True)
    files=sorted((SLIM/'p30/bench').glob('*.npz')); names=[f.stem for f in files if f.stem.startswith('train-')]
    train={}
    for key in names:
        f=np.load(SLIM/'p30/bench'/(key+'.npz'));train[key]=(features_at(f['adc'],f['tare']),f['mocap_b'])
    assert len(train)==8
    point=readj(SLIM/'p30/predictor.json');rows={'train':[],'validation':[],'test':[]}
    for f in files:
        key=f.stem;split=key.partition('-')[0];model=point
        if split=='train':
            xs,ys=zip(*[v for k,v in train.items() if k!=key]);model=fit_model(xs,ys)
        for kind,adc,tare,t,y in scenarios(key):
            b,X=predictors(adc,tare,t,model);err,ok=errors(b,y);rows[split].append((X,err,ok,kind))
    X,E,ok,_=zip(*rows['train']);X=np.vstack(X);E=np.vstack(E);ok=np.concatenate(ok)
    mean=X.mean(0);scale=np.maximum(X.std(0),.025);Z=design(X,mean,scale)
    w=[head(Z,E[:,0]),head(Z[ok],E[ok,1])]
    model=dict(center=mean.tolist(),scale=scale.tolist(),coef=np.array(w).tolist(),validation_scale=[1.,1.],train_motions=names)
    ratios=[[],[]]
    for X,e,eligible,_ in rows['validation']:
        v=var_eval(model,X);ratios[0].extend(e[:,0]**2/v[:,0]);ratios[1].extend(e[eligible,1]**2/v[eligible,1])
    model['validation_scale']=[float(np.mean(z)) for z in ratios];savej(OUT/'strain-variance.json',model)
    scores={}
    for kind in ('nominal','loop','offset','mixed'):
        ss=[r for r in rows['test'] if r[-1]==kind];X=np.vstack([r[0] for r in ss]);ee=np.vstack([r[1] for r in ss]);elig=np.concatenate([r[2] for r in ss]);v=var_eval(model,X)
        scores[kind]={}
        for j,label in enumerate(['theta','phi']):
            idx=elig if j else np.ones(len(X),bool);e=ee[idx,j];s=np.sqrt(v[idx,j])
            scores[kind][label]=dict(n=len(e),rmse_deg=float(np.mean(e**2)**.5),rms_sigma_deg=float(np.mean(s*s)**.5),in_95_fraction=float(np.mean(abs(e)<=1.96*s)))
    savej(OUT/'variance-bench.json',scores);print('holdout sampling coverage',[(k,round(v['theta']['in_95_fraction'],3)) for k,v in scores.items()],flush=True)
if __name__=='__main__':main()
