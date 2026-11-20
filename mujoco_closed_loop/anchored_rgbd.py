"""Causal metric RGB-D anchor graph. No simulator, scene, command or strain inputs."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

cv2.setNumThreads(1)
def inv(T):
    A=np.eye(4);A[:3,:3]=T[:3,:3].T;A[:3,3]=-A[:3,:3]@T[:3,3];return A
def difference(a,b):
    q=a@inv(b);return float(Rotation.from_matrix(q[:3,:3]).magnitude()),float(np.linalg.norm(q[:3,3]))
def average(poses):
    out=np.eye(4);out[:3,:3]=Rotation.from_matrix(np.stack([p[:3,:3] for p in poses])).mean().as_matrix()
    out[:3,3]=np.mean([p[:3,3] for p in poses],axis=0);return out

class AnchoredRGBD:
    def __init__(self,K,scale=1000.):
        self.K=K;self.scale=scale;self.orb=cv2.ORB_create(nfeatures=1800,fastThreshold=12)
        self.matcher=cv2.BFMatcher(cv2.NORM_HAMMING);self.bank=[];self.links={};self.initial=None
        self.last_added=-1e6;self.last_verified=-1e6;self.pending=[];self.last_pose=None;self.last_t=None
        self.last_quality=0.;self.last_diag={};self.count=0;self.plane_bank=[]
    def features(self,image,depth):
        # Pixel-depth mask rejects nearby moving robot surfaces without scene labels/CAD.
        z=depth.astype(np.float64)/self.scale;mask=((z>.55)&(z<7.5)).astype('uint8')*255
        keys,desc=self.orb.detectAndCompute(image,mask)
        if desc is None:return None
        xy=np.array([k.pt for k in keys]);uv=np.rint(xy).astype(int);zz=z[uv[:,1],uv[:,0]]
        good=(zz>.55)&(zz<7.5);xy=xy[good];zz=zz[good];desc=desc[good]
        if len(xy)<28:return None
        vec=(xy-self.K[:2,2])/np.diag(self.K)[:2]
        xyz=np.column_stack((vec*zz[:,None],zz))
        return (xy,xyz,desc,float(np.mean(mask>0)),self.planes(z))
    def planes(self,z):
        vv,uu=np.mgrid[2:z.shape[0]:5,2:z.shape[1]:5]
        zz=z[vv,uu].ravel();xy=np.c_[uu.ravel(),vv.ravel()]
        good=(zz>.55)&(zz<7.5);zz=zz[good];xy=xy[good]
        points=np.c_[(xy-self.K[:2,2])/np.diag(self.K)[:2]*zz[:,None],zz]
        original=len(points);out=[];rng=np.random.default_rng(9307)
        for order in range(4):
            if len(points)<80:break
            best=None;bestn=0
            for k in range(45):
                sample=points[rng.choice(len(points),3,replace=False)]
                n=np.cross(sample[1]-sample[0],sample[2]-sample[0]);size=np.linalg.norm(n)
                if size<1e-5:continue
                n/=size;keep=abs(points@n-sample[0]@n)<.008
                if keep.sum()>bestn:best,bestn=keep,int(keep.sum())
            if bestn<max(70,.025*original):break
            use=points[best];centre=use.mean(axis=0);_,ss,v=np.linalg.svd(use-centre,full_matrices=False)
            n=v[-1];res=float(np.sqrt(np.mean(((use-centre)@n)**2)))
            # Reject collinear support; retain narrow but metric two-dimensional planes.
            if ss[1]/max(ss[0],1e-10)>.02:out.append((n,float(bestn/max(1,original)),res))
            points=points[~best]
        return out
    def correct_planes(self,pose,feat,verified,t):
        if feat is None:return None
        Rmc=pose[:3,:3].T;matches=[];novel=[]
        for nc,w,e in feat[4]:
            pred=Rmc@nc
            choices=[(abs(float(pred@n)),n,weight) for n,weight in self.plane_bank]
            if choices and max(q[0] for q in choices)>np.cos(np.deg2rad(16)):
                _,ref,weight=max(choices,key=lambda x:x[0])
                if pred@ref<0:nc=-nc
                if not any(abs(ref@a[1])>.995 for a in matches):matches.append((nc,ref,w,e))
            else:novel.append((nc,w,e))
        if not matches:return None
        current=np.array([m[0] for m in matches]);target=np.array([m[1] for m in matches])
        weights=np.array([max(.05,m[2]) for m in matches])
        independent=len(matches)>1 and np.linalg.svd(target)[1][1]>.15
        if independent:
            fixed,_=Rotation.align_vectors(target,current,weights=weights);Rnew=fixed.as_matrix()
        else:
            a=Rmc@current[0];b=target[0];v=np.cross(a,b);c=float(np.clip(a@b,-1,1))
            K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
            Rnew=(np.eye(3)+K+K@K/max(1e-8,1+c))@Rmc
        angles=np.rad2deg(np.arccos(np.clip(np.sum((current@Rnew.T)*target,axis=1),-1,1)))
        if np.max(angles)>2.:return None
        result=pose.copy();result[:3,:3]=Rnew.T
        # Add a new orientation only when existing measured normals already fix two axes,
        # or a fresh multi-frame RGB-D correspondence registration supports the update.
        if (independent or verified) and len(self.plane_bank)<12:
            for nc,w,e in novel:
                n=Rnew@nc
                if not any(abs(n@a)>.98 for a,_ in self.plane_bank):self.plane_bank.append((n,w))
        information=sum(w*(np.eye(3)-np.outer(n,n)) for (_,n,w,_) in matches)
        values,vectors=np.linalg.eigh(information)
        sd=np.deg2rad(.65+.25/np.sqrt(np.maximum(values,.025)))
        covariance=(vectors*sd**2)@vectors.T
        return result,dict(plane_count=len(matches),plane_coverage=float(sum(m[2] for m in matches)),
             plane_fit_m=float(max(m[3] for m in matches)),plane_residual_deg=float(max(angles)),
             plane_rank=int(np.sum(values>.03)),cov_map=covariance.tolist(),
             plane_bank=len(self.plane_bank))
    def fit(self,ref,cur):
        _,xyz,rd,_=ref[:4];xy,cxyz,cd,depth_ratio=cur[:4]
        forward=self.matcher.knnMatch(rd,cd,k=2);back=self.matcher.match(cd,rd)
        rev={m.queryIdx:m.trainIdx for m in back}
        pairs=[(m.queryIdx,m.trainIdx) for matches in forward if len(matches)==2 for m,n in [matches]
               if m.distance<.77*n.distance and rev.get(m.trainIdx)==m.queryIdx]
        if len(pairs)<25:return None
        ia,ib=np.asarray(pairs).T;p=xyz[ia];q=xy[ib]
        ok,r,v,inl=cv2.solvePnPRansac(p,q,self.K,None,iterationsCount=80,reprojectionError=2.5,
                                   confidence=.995,flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl)<22 or len(inl)/len(p)<.5:return None
        ix=inl.ravel()
        r,v=cv2.solvePnPRefineLM(p[ix],q[ix],self.K,None,r,v)
        R=cv2.Rodrigues(r)[0];pred=p@R.T+v.ravel()
        projected=pred@self.K.T;projected=projected[:,:2]/projected[:,2,None]
        err=np.linalg.norm(projected-q,axis=1)
        dz=abs(pred[:,2]-cxyz[ib,2]);good=(err<2.5)&(dz<.035)&(pred[:,2]>.2)
        if good.sum()<22 or good.mean()<.45:return None
        r,v=cv2.solvePnPRefineLM(p[good],q[good],self.K,None,r,v)
        R=cv2.Rodrigues(r)[0];pred=p[good]@R.T+v.ravel()
        projected=pred@self.K.T;projected=projected[:,:2]/projected[:,2,None]
        med=float(np.median(np.linalg.norm(projected-q[good],axis=1)))
        deptherr=float(np.median(abs(pred[:,2]-cxyz[ib[good],2])))
        extent=np.percentile(q[good],95,axis=0)-np.percentile(q[good],5,axis=0)
        if med>1.7 or deptherr>.027 or np.any(extent<[75,48]):return None
        T=np.eye(4);T[:3,:3]=R;T[:3,3]=v.ravel()
        return T,dict(inliers=int(good.sum()),ratio=float(good.mean()),reproj=med,
                      depth_error=deptherr,extent=extent.tolist(),depth_ratio=depth_ratio)
    def recover(self,cur,t):
        # Early anchors have priority; established intermediate anchors extend the field of view.
        early=[e for e in self.bank if e['t']<14.]
        rest=[e for e in self.bank if e['t']>=14.]
        order=early[::max(1,len(early)//4)]+rest[-6:]+rest[::max(1,len(rest)//8)]
        seen=set();fits=[]
        for key in order:
            if key['t'] in seen:continue
            seen.add(key['t']);fit=self.fit(key['feature'],cur)
            if fit is None:continue
            rel,diag=fit;fits.append((rel@key['pose'],key,diag))
            if len(fits)>=5:break
        best=[]
        for a in fits:
            cluster=[q for q in fits if difference(a[0],q[0])[0]<np.deg2rad(1.4)
                     and difference(a[0],q[0])[1]<.045]
            if len(cluster)>len(best):best=cluster
        if len(best)<2 or max(q[1]['t'] for q in best)-min(q[1]['t'] for q in best)<.20:return None
        T=average([q[0] for q in best])
        disagreement=max(difference(T,q[0])[0] for q in best)
        diag={k:float(np.mean([q[2][k] for q in best])) for k in ['inliers','ratio','reproj','depth_error','depth_ratio']}
        diag['extent']=np.min([q[2]['extent'] for q in best],axis=0).tolist()
        diag.update(anchor_count=len(best),anchor_times=[q[1]['t'] for q in best],
                    anchor_min=min(q[1]['t'] for q in best),agreement_deg=float(np.rad2deg(disagreement)))
        return T,diag
    def update(self,t,native,mid,native_valid,feature,frame):
        self.count+=1;diag=dict(t=float(t),frame=int(frame),map=int(mid),native_valid=bool(native_valid))
        if self.initial is None and native_valid and feature is not None:
            self.initial=(t,native.copy(),mid);self.links[mid]=np.eye(4)
            self.bank.append(dict(t=t,pose=native.copy(),feature=feature));self.last_added=t
            self.last_verified=t
            for normal,weight,error in feature[4]:self.plane_bank.append((native[:3,:3].T@normal,weight))
        if self.initial is None or not native_valid:
            self.pending=[];diag.update(valid=False,reason='native_lost');return None,0.,diag
        if mid==self.initial[2] and t<14. and feature is not None and t-self.last_added>.24:
            self.bank.append(dict(t=t,pose=native.copy(),feature=feature));self.last_added=t
        fit=self.recover(feature,t) if feature is not None and (frame%5==0 or (mid not in self.links and frame%2==0)) else None
        checked=False;relock=False
        if fit is not None:
            optical,geom=fit;candidate=inv(native)@optical;checked=True
            if mid not in self.links:
                if self.pending and (mid!=self.pending[-1][0] or difference(candidate,self.pending[-1][1])[0]>np.deg2rad(1.4) or difference(candidate,self.pending[-1][1])[1]>.045):
                    self.pending=[]
                self.pending.append((mid,candidate))
                if len(self.pending)>=3:self.links[mid]=candidate;relock=True;self.pending=[]
            else:
                # Forward-only correction; previous outputs are never revisited.
                self.links[mid]=candidate
            if mid in self.links:
                self.last_verified=t;self.last_diag=geom.copy()
                if t>=14 and t-self.last_added>=.60 and geom['inliers']>=40 and geom['agreement_deg']<.8:
                    self.bank.append(dict(t=t,pose=optical.copy(),feature=feature));self.last_added=t
        if mid not in self.links:
            diag.update(valid=False,reason='unaligned_submap');return None,0.,diag
        corrected=native@self.links[mid]
        plane_info=self.correct_planes(corrected,feature,checked,t)
        if plane_info is not None:
            corrected,plane_diag=plane_info;self.links[mid]=inv(native)@corrected
            self.last_verified=t
            if t>=14 and t-self.last_added>=.60 and plane_diag['plane_rank']==3 and plane_diag['plane_coverage']>.35:
                self.bank.append(dict(t=t,pose=corrected.copy(),feature=feature));self.last_added=t
        else:plane_diag={}
        age=max(0.,t-self.last_verified)
        g=self.last_diag.copy();g.setdefault('inliers',50.);g.setdefault('ratio',.5);g.setdefault('reproj',1.2)
        g.setdefault('depth_ratio',feature[3] if feature is not None else 0.);g.setdefault('extent',[100.,70.])
        g.setdefault('agreement_deg',.5);g.setdefault('anchor_count',0)
        if plane_diag:
            g.update(plane_diag)
            g['depth_ratio']=feature[3]
        rate=0.
        if self.last_pose is not None and self.last_t is not None:
            rate=np.rad2deg(difference(corrected,self.last_pose)[0])/max(.01,t-self.last_t)
        self.last_pose=corrected.copy();self.last_t=t
        # Observable geometry controls confidence; no error-to-truth or command terms.
        coverage=min(1.,min(np.array(g['extent'])/np.array([260.,180.])))
        q=np.clip(g['inliers']/180.,.08,1.)*np.clip(g['ratio']/.8,.1,1.)*coverage
        q*=np.exp(-.5*(g['reproj']/1.4)**2)*np.sqrt(max(.01,g['depth_ratio']))
        q*=np.exp(-age/4.)/(1.+(rate/110.)**2)
        sigma=.65+1.15*(1.-q)+.5*g['agreement_deg']+.10*min(age,10.)+.002*rate
        if plane_diag:
            q=max(q,min(.9,plane_diag['plane_coverage']*2.5)/(1.+(rate/110.)**2))
            sigma=.70+.4*(1-q)+.003*rate
        valid=bool(feature is not None and age<6. and rate<160.)
        diag.update(g);diag.update(checked=checked,relocked=relock,age_s=age,angular_rate_deg_s=rate,
                       quality=float(q) if valid else 0.,sigma_deg=float(sigma),valid=valid,
                       reason='anchored' if valid else 'observable_geometry_degraded',keyframes=len(self.bank))
        return corrected if valid else None,float(q) if valid else 0.,diag

