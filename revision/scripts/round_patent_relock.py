"""Connect a newly tracked local map to the original map using only synchronized wrist RGB-D."""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from round_patent_camera import CASES,OLD,INPUT,OUT,images
from round_wrist_fusion import savej
cv2.setNumThreads(1)
def transform(row):
    m=np.eye(4);m[:3]=np.asarray([getattr(row,'r'+str(i)) for i in range(12)]).reshape(3,4);return m
def delta(a,b):
    rel=a@np.linalg.inv(b)
    return float(np.rad2deg(Rotation.from_matrix(rel[:3,:3]).magnitude())),float(np.linalg.norm(rel[:3,3]))
class Keyframes:
    def __init__(self,settings):
        fs=cv2.FileStorage(str(settings),cv2.FILE_STORAGE_READ)
        f=lambda k:fs.getNode(k).real()
        self.K=np.array([[f('Camera1.fx'),0,f('Camera1.cx')],[0,f('Camera1.fy'),f('Camera1.cy')],[0,0,1.]])
        self.scale=f('RGBD.DepthMapFactor');fs.release();self.orb=cv2.ORB_create(nfeatures=950)
        self.match=cv2.BFMatcher(cv2.NORM_HAMMING);self.bank=[]
    def features(self,im,depth):
        keys,desc=self.orb.detectAndCompute(im,None)
        if desc is None:return None
        xy=np.array([k.pt for k in keys]);uv=np.rint(xy).astype(int)
        z=depth[uv[:,1],uv[:,0]]/self.scale;good=(z>.12)&(z<6)
        if good.sum()<28:return None
        xy,z=xy[good],z[good];v=(xy-self.K[:2,2])/np.diag(self.K)[:2]
        xyz=np.column_stack((v*z[:,None],z));return xy,xyz,desc[good]
    def try_pair(self,ref,cur):
        _,xyz,rd=ref;xy,cur_xyz,cd=cur
        forward=self.match.knnMatch(rd,cd,k=2);back=self.match.match(cd,rd)
        rev={m.queryIdx:m.trainIdx for m in back}
        pairs=[(m.queryIdx,m.trainIdx) for match in forward if len(match)==2 for m,n in [match]
               if m.distance<.75*n.distance and rev.get(m.trainIdx)==m.queryIdx]
        if len(pairs)<35:return None
        i,j=np.asarray(pairs).T;p=xyz[i];x=xy[j]
        ok,r,v,inl=cv2.solvePnPRansac(p,x,self.K,None,iterationsCount=100,reprojectionError=2.5,confidence=.995,flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl)<28 or len(inl)/len(i)<.55:return None
        ix=inl.ravel();M=np.eye(4);M[:3,:3]=cv2.Rodrigues(r)[0];M[:3,3]=v[:,0]
        pred=p[ix]@M[:3,:3].T+M[:3,3];uv=pred@self.K.T;uv=uv[:,:2]/uv[:,2,None]
        err=np.median(np.linalg.norm(uv-x[ix],axis=1));dz=np.median(abs(pred[:,2]-cur_xyz[j[ix],2]))
        extent=np.percentile(x[ix],95,axis=0)-np.percentile(x[ix],5,axis=0)
        if err>1.5 or dz>.035 or (pred[:,2]<=0).any() or np.any(extent<[70,45]):return None
        return M,len(ix),float(err),float(dz)
    def recover(self,cur):
        matches=[]
        for t,old,feat in reversed(self.bank):
            fit=self.try_pair(feat,cur)
            if fit is None:continue
            rel,n,e,d=fit;matches.append((rel@old,t,n,e,d))
            if len(matches)>=3:break
        if len(matches)<2:return None
        for i,(T,t,n,e,d) in enumerate(matches):
            for U,other,_,_,_ in matches[i+1:]:
                ang,dist=delta(T,U)
                if abs(t-other)>=.25 and ang<2 and dist<.04:return T,n,e,d
        return None
def process(case,cond):
    dest=OUT/'camera'/case;dest.mkdir(parents=True,exist_ok=True)
    csv=dest/'graded.csv' if cond=='graded' else OLD/case/(cond+'.csv')
    tab=pd.read_csv(csv);raw=list(images(case));assert np.allclose(tab.t,[v[0] for v in raw],atol=1e-6)
    reader=Keyframes(INPUT/case/'settings.yaml');cv2.setRNGSeed(7101)
    frames=tab.copy();cols=['r'+str(i) for i in range(12)]
    anchor=None;first={};last_bank=-10.;pending=[];link={};seen={};gain=[];original=[];bridged=[];tested=0;null_count=0
    for i,row in enumerate(tab.itertuples()):
        ok=row.state==2 and row.matched>=15 and np.isfinite(row.reproj_px) and row.reproj_px<=3 and row.map_id>=0
        if anchor is None and ok and row.t<13.:anchor=int(row.map_id)
        mid=int(row.map_id);good=bool(ok and anchor is not None and mid==anchor)
        original.append(good);new=bool(ok and anchor is not None and mid!=anchor)
        def observe():
            _,rp,dp=raw[i]
            if cond=='graded':
                # Select the identical pixels materialized for ORB-SLAM3.
                p=OUT/'inputs'/case;alter=p/('r%04d.png'%i)
                rp=alter if alter.is_file() else p/rp.name;dp=p/dp.name
            if cond=='occluded' and bool(row.occluded):return None
            return reader.features(cv2.imread(str(rp),0),cv2.imread(str(dp),cv2.IMREAD_UNCHANGED))
        if good and row.t-last_bank>=.48 and row.matched>=60:
            feat=observe()
            if feat is not None:
                reader.bank.append((row.t,transform(row),feat));reader.bank=reader.bank[-16:];last_bank=row.t
        if not new:pending=[]
        if new and mid not in link and i%3==0:
            cur=observe();tested+=1
            if cur is not None and reader.bank:
                # Negative control: a geometry-inconsistent permutation of current pixels.
                xy,xyz,desc=cur;fake=(xy[::-1].copy(),xyz[::-1].copy(),desc)
                if reader.try_pair(reader.bank[-1][2],fake) is not None:null_count+=1
            bridge=reader.recover(cur) if cur is not None else None
            if bridge is None:pending=[]
            else:
                T,n,e,d=bridge;candidate=np.linalg.inv(transform(row))@T
                if pending:
                    ang,dist=delta(candidate,pending[-1][1])
                    if pending[-1][0]!=mid or ang>2. or dist>.04:pending=[]
                pending.append((mid,candidate))
                if len(pending)>=3:
                    link[mid]=candidate;first[mid]=float(row.t)
                    gain.append(dict(map=mid,capture_s=row.t,inliers=n,median_px=e,depth_disagreement_m=d));pending=[]
        use=new and mid in link
        if use:
            T=transform(row)@link[mid];frames.loc[i,cols]=T[:3].reshape(-1);frames.loc[i,'map_id']=anchor
        bridged.append(bool(use))
    frames.to_csv(dest/(cond+'-registered.csv'),index=False)
    savej(dest/(cond+'-registration.json'),dict(initial_map=anchor,connections=gain,native_anchored=int(sum(original)),
        verified_additional=int(sum(bridged)),recovery_tests=tested,negative_geometry_acceptances=null_count))
    np.savez_compressed(dest/(cond+'-map.npz'),t=tab.t,native=original,bridged=bridged)
    assert null_count==0
    print(case,cond,'native',sum(original),'added',sum(bridged),'connections',gain,flush=True)
if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('conditions',nargs='+');o=a.parse_args()
    for cond in o.conditions:
        for case in CASES:process(case,cond)
