"""Controlled surveyed ArUco map. Runtime localization consumes RGB pixels only."""
import cv2,numpy as np
from xml.etree.ElementTree import SubElement as S
from visual_odometry import VisualEstimate

DICTIONARY=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

def layout():
    items=[];identifier=0
    for x in [-1.4,-.7,0.,.7,1.4]:
        for y in [-1.6,-.8,0.,.8,1.6]:
            items.append((identifier,np.array([x,y,.004]),np.array([1.,0,0]),np.array([0,-1.,0]),.30));identifier+=1
    for y in [-.8,0.,.8]:
        items.append((identifier,np.array([-1.05,y,.004]),np.array([1.,0,0]),np.array([0,-1.,0]),.20));identifier+=1
    for side in [-1,1]:
        for y in [-.6,0.,.6]:
            # Pixel down is world -Z; orientation faces the room interior.
            items.append((identifier,np.array([side*1.552,y,1.35]),np.array([0,-side,0.]),np.array([0,0,-1.]),.40 if y==0 else .55));identifier+=1
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
    def __init__(self,T_initial):
        self.pose=T_initial.copy();self.points=marker_corners();self.params=cv2.aruco.DetectorParameters_create();self.params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    def update(self,frame):
        corners,ids,_=cv2.aruco.detectMarkers(frame.rgb,DICTIONARY,parameters=self.params)
        objects=[];pixels=[];surface_W=[];surface_C=[]
        if ids is not None:
            for c,i in zip(corners,ids.ravel()):
                if int(i) in self.points:
                    world=self.points[int(i)];image=c.reshape(4,2);objects.extend(world);pixels.extend(image)
                    H=cv2.getPerspectiveTransform(np.array([[0,0],[1,0],[1,1],[0,1]],np.float32),image.astype(np.float32))
                    for y in np.linspace(.1,.9,5):
                        for x in np.linspace(.1,.9,5):
                            uv=H@np.array([x,y,1.]);uv=uv[:2]/uv[2];ix,iy=np.rint(uv).astype(int)
                            if not (0<=iy<frame.depth_m.shape[0] and 0<=ix<frame.depth_m.shape[1]):continue
                            z=frame.depth_m[iy,ix]
                            if not np.isfinite(z):continue
                            surface_W.append(world[0]+x*(world[1]-world[0])+y*(world[3]-world[0]))
                            surface_C.append([(uv[0]-frame.K[0,2])*z/frame.K[0,0],(uv[1]-frame.K[1,2])*z/frame.K[1,1],z])
        if len(objects)<4:return VisualEstimate(frame.stamp,self.pose.copy(),False,0,float('inf'),'fiducials_not_visible_hold')
        obj=np.asarray(objects,dtype=np.float64);img=np.asarray(pixels,dtype=np.float64)
        success,r,t=cv2.solvePnP(obj,img,frame.K,None,flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:return VisualEstimate(frame.stamp,self.pose.copy(),False,0,float('inf'),'fiducial_pnp_failed_hold')
        reprojected,_=cv2.projectPoints(obj,r,t,frame.K,None)
        error=float(np.sqrt(np.mean((reprojected.reshape(-1,2)-img)**2)))
        reason='surveyed_fiducial_rgb_pnp'
        if len(surface_W)>=12:
            W=np.array(surface_W);C=np.array(surface_C);rng=np.random.default_rng(0);best=None
            def rigid(A,B):
                ac=A.mean(axis=0);bc=B.mean(axis=0);u,_,v=np.linalg.svd((A-ac).T@(B-bc));Q=v.T@u.T
                if np.linalg.det(Q)<0:v[-1]*=-1;Q=v.T@u.T
                return Q,bc-Q@ac
            for _ in range(48):
                ids=rng.choice(len(W),3,replace=False)
                if np.linalg.norm(np.cross(W[ids[1]]-W[ids[0]],W[ids[2]]-W[ids[0]]))<1e-6:continue
                Q,b=rigid(W[ids],C[ids]);inliers=np.linalg.norm(W@Q.T+b-C,axis=1)<.008
                if best is None or inliers.sum()>best.sum():best=inliers
            if best is not None and best.sum()>=max(12,.60*len(W)):
                R,b=rigid(W[best],C[best]);depth_t=b.reshape(3,1);depth_r=cv2.Rodrigues(R)[0]
                projected,_=cv2.projectPoints(obj,depth_r,depth_t,frame.K,None)
                depth_error=float(np.sqrt(np.mean((projected.reshape(-1,2)-img)**2)))
                if depth_error<=1.5:
                    r,t,error=depth_r,depth_t,depth_error;reason='surveyed_fiducial_rgbd_ransac'
        if error>1.5:return VisualEstimate(frame.stamp,self.pose.copy(),False,len(obj),error,'fiducial_reprojection_hold')
        CW=np.eye(4);CW[:3,:3]=cv2.Rodrigues(r)[0];CW[:3,3]=t.ravel();self.pose=np.linalg.inv(CW)
        return VisualEstimate(frame.stamp,self.pose.copy(),True,len(obj),error,reason)
