"""Controlled surveyed ArUco prior; runtime localization consumes RGB-D."""
import cv2,numpy as np
from xml.etree.ElementTree import SubElement as S
from visual_odometry import VisualEstimate

DICTIONARY=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

def _rigid_fit(W,C):
    wc=W.mean(axis=0);cc=C.mean(axis=0);u,_,v=np.linalg.svd((W-wc).T@(C-cc));R=v.T@u.T
    if np.linalg.det(R)<0:v[-1]*=-1;R=v.T@u.T
    return R,cc-R@wc

def _robust_rigid_fit(W,C):
    R,t=_rigid_fit(W,C);residual=np.linalg.norm(W@R.T+t-C,axis=1)
    inliers=residual<.010
    if np.percentile(residual,95)>.008:
        rng=np.random.default_rng(0);best=inliers.copy();best_score=(int(best.sum()),-float(np.minimum(residual,.01).sum()))
        for _ in range(64):
            ids=rng.choice(len(W),3,replace=False);v=W[ids]
            if np.linalg.norm(np.cross(v[1]-v[0],v[2]-v[0]))<1e-8:continue
            r,q=_rigid_fit(W[ids],C[ids]);err=np.linalg.norm(W@r.T+q-C,axis=1);mask=err<.010
            score=(int(mask.sum()),-float(np.minimum(err,.01).sum()))
            if score>best_score:best,best_score=mask,score
        inliers=best
    if inliers.sum()<12 or inliers.mean()<.60:return None
    R,t=_rigid_fit(W[inliers],C[inliers]);rms=float(np.sqrt(np.mean(np.sum((W[inliers]@R.T+t-C[inliers])**2,axis=1))))
    if rms>.006:return None
    return R,t,rms,int(inliers.sum())

def layout():
    items=[];identifier=0
    for x in [-.7,0.,.7,1.4]:
        for y in [-1.6,-.9,-.2,.5]:
            items.append((identifier,np.array([x,y,.004]),np.array([1.,0,0]),np.array([0,-1.,0]),.30));identifier+=1
    for side in [-1,1]:
        for y in [-.6,0.,.6]:
            # Pixel down is world -Z; orientation faces the room interior.
            items.append((identifier,np.array([side*1.552,y,1.35]),np.array([0,-side,0.]),np.array([0,0,-1.]),.40 if y==0 else .55));identifier+=1
    # Entry/exit chamber boards remain visible when the portal occludes the
    # central wall boards. Append IDs so all earlier surveyed IDs are stable.
    for side in [-1,1]:
        for y in [-1.4,1.4]:
            items.append((identifier,np.array([side*1.552,y,1.35]),np.array([0,-side,0.]),np.array([0,0,-1.]),.45));identifier+=1
    for side in [-1,1]:
        for z in [.70,2.0]:
            for y in [-1.4,-.6,.6,1.4]:
                items.append((identifier,np.array([side*1.552,y,z]),np.array([0,-side,0.]),np.array([0,0,-1.]),.35));identifier+=1
    for y in [-2.2,1.2]:
        for x in [-.7,0.,.7,1.4]:
            items.append((identifier,np.array([x,y,.004]),np.array([1.,0,0]),np.array([0,-1.,0]),.30));identifier+=1
    assert identifier==50
    return items

def marker_corners():
    return {i:np.array([p+(-u-v)*L/2,p+(u-v)*L/2,p+(u+v)*L/2,p+(-u+v)*L/2]) for i,p,u,v,L in layout()}

def add_fiducials(root):
    w=root.find('worldbody')
    for i,p,u,v,L in layout():
        n=np.cross(u,v);rotation=np.column_stack([u,v,n])
        xy=' '.join(map(str,np.r_[u,v]));bitmap=cv2.aruco.drawMarker(DICTIONARY,i,6)
        # Actual printed cells; no texture UV assumptions or marker IDs in sensor packets.
        S(w,'geom',type='box',pos=' '.join(map(str,p+n*.0008)),xyaxes=xy,size=f'{L*.64} {L*.64} .0005',rgba='1 1 1 1',contype='0',conaffinity='0')
        for row in range(6):
            for col in range(6):
                if bitmap[row,col]>0:continue
                center=p+u*((col+.5-3)*L/6)+v*((row+.5-3)*L/6)
                S(w,'geom',type='box',pos=' '.join(map(str,center)),xyaxes=xy,size=f'{L/12} {L/12} .0005',rgba='.01 .01 .01 1',contype='0',conaffinity='0')

class FiducialLocalization:
    def __init__(self,T_initial,refine=False,robust_depth=False):
        self.pose=T_initial.copy();self.points=marker_corners();self.params=cv2.aruco.DetectorParameters_create();self.params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
        self.refine=refine
        self.robust_depth=robust_depth
        grid=np.linspace(.1,.9,5);self.samples=np.stack(np.meshgrid(grid,grid),axis=-1).reshape(-1,2)
    def update(self,frame):
        corners,ids,_=cv2.aruco.detectMarkers(frame.rgb,DICTIONARY,parameters=self.params)
        objects=[];pixels=[];surface_W=[];surface_C=[]
        if ids is not None:
            for c,i in zip(corners,ids.ravel()):
                if int(i) in self.points:
                    world=self.points[int(i)];image=c.reshape(4,2)
                    if self.robust_depth and (np.linalg.norm(image-np.roll(image,1,axis=0),axis=1).min()<12 or abs(cv2.contourArea(image.astype(np.float32)))<200):continue
                    objects.extend(world);pixels.extend(image)
                    H=cv2.getPerspectiveTransform(np.array([[0,0],[1,0],[1,1],[0,1]],np.float32),image.astype(np.float32))
                    xy=self.samples;uvh=np.column_stack([xy,np.ones(len(xy))])@H.T;uv=uvh[:,:2]/uvh[:,2:]
                    indices=np.floor(uv).astype(int);ix,iy=indices.T
                    valid=(iy>=0)&(iy<frame.depth_m.shape[0]-1)&(ix>=0)&(ix<frame.depth_m.shape[1]-1)
                    xy,uv,indices=xy[valid],uv[valid],indices[valid];ix,iy=indices.T
                    depth=np.column_stack([frame.depth_m[iy,ix],frame.depth_m[iy,ix+1],frame.depth_m[iy+1,ix],frame.depth_m[iy+1,ix+1]])
                    valid=np.isfinite(depth).all(axis=1);xy,uv,indices,depth=xy[valid],uv[valid],indices[valid],depth[valid]
                    fx,fy=(uv-indices).T;weights=np.column_stack([(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy])
                    # Inverse optical depth is affine on a planar marker.
                    z=1./np.sum(weights/depth,axis=1)
                    surface_W.extend(world[0]+xy[:,0,None]*(world[1]-world[0])+xy[:,1,None]*(world[3]-world[0]))
                    surface_C.extend(np.column_stack([(uv[:,0]-frame.K[0,2])*z/frame.K[0,0],(uv[:,1]-frame.K[1,2])*z/frame.K[1,1],z]))
        if len(objects)<4:return VisualEstimate(frame.stamp,self.pose.copy(),False,0,float('inf'),'fiducials_not_visible_hold')
        obj=np.asarray(objects,dtype=np.float64);img=np.asarray(pixels,dtype=np.float64)
        if len(surface_W)>=12:
            W=np.array(surface_W);C=np.array(surface_C)
            if self.robust_depth:fit=_robust_rigid_fit(W,C)
            else:
                R,t=_rigid_fit(W,C);fit=(R,t,float(np.sqrt(np.mean(np.sum((W@R.T+t-C)**2,axis=1)))),len(W))
            if fit is None:return VisualEstimate(frame.stamp,self.pose.copy(),False,0,float('inf'),'fiducial_metric_depth_inconsistent_hold')
            R,t,metric_rms,depth_inliers=fit;t=t.reshape(3,1);r=cv2.Rodrigues(R)[0];success=True
        else:return VisualEstimate(frame.stamp,self.pose.copy(),False,0,float('inf'),'fiducial_metric_depth_insufficient_hold')
        if not success:return VisualEstimate(frame.stamp,self.pose.copy(),False,0,float('inf'),'fiducial_pnp_failed_hold')
        if self.refine and len(surface_W)>=12:
            # Depth gives metric initialization; fit the observed subpixel
            # marker corners to avoid foreshortened depth-plane normal bias.
            refined,rr,tt=cv2.solvePnP(obj,img,frame.K,None,r,t,True,flags=cv2.SOLVEPNP_ITERATIVE)
            if refined:r,t=rr,tt
        reprojected,_=cv2.projectPoints(obj,r,t,frame.K,None);error=float(np.sqrt(np.mean((reprojected.reshape(-1,2)-img)**2)))
        if error>1.5:return VisualEstimate(frame.stamp,self.pose.copy(),False,len(obj),error,'fiducial_reprojection_hold')
        CW=np.eye(4);CW[:3,:3]=cv2.Rodrigues(r)[0];CW[:3,3]=t.ravel();self.pose=np.linalg.inv(CW)
        result=VisualEstimate(frame.stamp,self.pose.copy(),True,depth_inliers,error,'surveyed_fiducial_localization')
        result.depth_fit_rms_m=metric_rms
        return result
