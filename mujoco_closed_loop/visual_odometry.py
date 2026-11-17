"""RGB-D feature odometry: depth lifting + ORB + RANSAC PnP, no simulator import."""
import cv2,numpy as np
from dataclasses import dataclass

@dataclass
class VisualEstimate:
    stamp:float
    T_WC:np.ndarray
    valid:bool
    inliers:int
    reprojection_px:float
    reason:str

class RGBDOdometry:
    def __init__(self,T_WC_initial):
        self.pose=np.array(T_WC_initial,copy=True);self.previous=None;self.reference_pose=self.pose.copy()
        self.orb=cv2.ORB_create(nfeatures=1800,fastThreshold=10)
        self.matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
        self.keyframes=[];self.last_valid_stamp=0.;self.frames=0
    def update(self,frame):
        result=self._track(frame)
        self.frames+=1
        if result.valid and self.keyframes and self.frames%10==0:
            saved=(self.previous,self.reference_pose.copy(),self.pose.copy())
            self.previous,self.reference_pose=self.keyframes[0][0],self.keyframes[0][1].copy()
            anchored=self._track(frame)
            if anchored.valid and anchored.inliers>=100 and np.linalg.norm(anchored.T_WC[:3,3]-saved[2][:3,3])<.15:
                anchored.reason='initial_keyframe_loop_anchor';result=anchored
            else:self.previous,self.reference_pose,self.pose=saved
        if not result.valid and self.previous is not None:
            saved=(self.previous,self.reference_pose.copy(),self.pose.copy())
            for previous,reference in reversed(self.keyframes):
                if previous is saved[0]:continue
                self.previous=previous;self.reference_pose=reference.copy()
                candidate=self._track(frame)
                if candidate.valid:
                    candidate.reason='relocalised_keyframe'
                    result=candidate;break
            if not result.valid:self.previous,self.reference_pose,self.pose=saved
        if result.valid and self.previous is not None and not any(k[0] is self.previous for k in self.keyframes):
            self.keyframes.append((self.previous,self.reference_pose.copy()))
            if len(self.keyframes)>16:self.keyframes.pop(1)  # retain initial anchor
        if result.valid:self.last_valid_stamp=frame.stamp
        return result

    def _track(self,frame):
        gray=cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2GRAY);kp,desc=self.orb.detectAndCompute(gray,getattr(frame,'feature_mask',None))
        if self.previous is None:
            if desc is not None and len(kp)>=30:
                self.previous=(frame,kp,desc)
                return VisualEstimate(frame.stamp,self.pose.copy(),True,len(kp),0.,'initialised_known_pose')
            return VisualEstimate(frame.stamp,self.pose.copy(),False,0,float('inf'),'insufficient_texture')
        if desc is None:return VisualEstimate(frame.stamp,self.pose.copy(),False,0,float('inf'),'no_features_hold')
        old,okp,od=self.previous;matches=self.matcher.knnMatch(od,desc,k=2);points=[];pixels=[]
        for pair in matches:
            if len(pair)!=2:continue
            a,b=pair
            if a.distance>=.7*b.distance:continue
            x,y=okp[a.queryIdx].pt;ix,iy=int(round(x)),int(round(y));z=old.depth_m[iy,ix]
            if not np.isfinite(z):continue
            points.append([(x-old.K[0,2])*z/old.K[0,0],(y-old.K[1,2])*z/old.K[1,1],z]);pixels.append(kp[a.trainIdx].pt)
        if len(points)<12:return VisualEstimate(frame.stamp,self.pose.copy(),False,len(points),float('inf'),'few_depth_matches_hold')
        points=np.asarray(points,dtype=np.float64);pixels=np.asarray(pixels,dtype=np.float64)
        success,r,t,indices=cv2.solvePnPRansac(points,pixels,frame.K,None,iterationsCount=150,reprojectionError=2.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
        if not success or indices is None or len(indices)<12:
            return VisualEstimate(frame.stamp,self.pose.copy(),False,0,float('inf'),'pnp_failed_hold')
        selected=indices.ravel()
        r,t=cv2.solvePnPRefineLM(points[selected],pixels[selected],frame.K,None,r,t)
        projected,_=cv2.projectPoints(points[selected],r,t,frame.K,None)
        residual=float(np.sqrt(np.mean((projected.reshape(-1,2)-pixels[selected])**2)))
        if residual>1.5:
            return VisualEstimate(frame.stamp,self.pose.copy(),False,len(selected),residual,'reprojection_failed_hold')
        relative=np.eye(4);relative[:3,:3]=cv2.Rodrigues(r)[0];relative[:3,3]=t.ravel()
        predicted=points[selected]@relative[:3,:3].T+relative[:3,3]
        uv=np.rint(pixels[selected]).astype(int);depth=frame.depth_m[uv[:,1],uv[:,0]]
        consistent=np.isfinite(depth)&(predicted[:,2]>0)&(np.abs(predicted[:,2]-depth)<(.05+.02*depth))
        if consistent.sum()<max(12,.6*len(selected)):
            return VisualEstimate(frame.stamp,self.pose.copy(),False,int(consistent.sum()),residual,'depth_inconsistent_hold')
        candidate=self.reference_pose@np.linalg.inv(relative)
        elapsed=max(.01,frame.stamp-self.last_valid_stamp)
        jump=np.linalg.norm(candidate[:3,3]-self.pose[:3,3])
        angle=np.linalg.norm(cv2.Rodrigues(candidate[:3,:3]@self.pose[:3,:3].T)[0])
        if not np.isfinite(candidate).all() or jump>.10+2*elapsed or angle>np.deg2rad(15)+2*elapsed:
            return VisualEstimate(frame.stamp,self.pose.copy(),False,len(selected),residual,'motion_inconsistent_hold')
        self.pose=candidate
        # Anchor overlapping views to a retained RGB-D keyframe to limit drift.
        if len(selected)<150 or np.linalg.norm(relative[:3,3])>.3 or np.linalg.norm(r)>.4:
            self.previous=(frame,kp,desc);self.reference_pose=self.pose.copy()
        return VisualEstimate(frame.stamp,self.pose.copy(),True,len(selected),residual,'tracking')

def detect_red_target(frame):
    hsv=cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2HSV)
    mask=(((hsv[:,:,0]<12)|(hsv[:,:,0]>170))&(hsv[:,:,1]>140)&(hsv[:,:,2]>60)).astype(np.uint8)
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
    if n<=1:return None
    label=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);ys,xs=np.where(labels==label);z=frame.depth_m[ys,xs];good=np.isfinite(z)
    if good.sum()<12:return None
    zs=z[good];xs=xs[good];ys=ys[good]
    xyz=np.column_stack([(xs-frame.K[0,2])*zs/frame.K[0,0],(ys-frame.K[1,2])*zs/frame.K[1,1],zs])
    return dict(surface_center_C=np.median(xyz,axis=0),pixels=int(good.sum()),depth_valid_fraction=float(good.mean()))

LAST_TARGET_DIAGNOSTIC={}

def detect_supported_cylinder(frame,T_WC,radius=.012,height=.04):
    """Controlled upright cylinder on a locally observed horizontal support plane.

    Estimates centre from RGB-D geometry, not simulator object coordinates.
    """
    global LAST_TARGET_DIAGNOSTIC
    LAST_TARGET_DIAGNOSTIC={'reason':'started'}
    def reject(reason):
        LAST_TARGET_DIAGNOSTIC['reason']=reason
        return None
    from scipy.optimize import least_squares
    hsv=cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2HSV)
    red=(((hsv[:,:,0]<12)|(hsv[:,:,0]>170))&(hsv[:,:,1]>140)&(hsv[:,:,2]>60)).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(red)
    if n<=1:return reject('no_red_component')
    label=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));mask=(labels==label).astype(np.uint8)
    mask=cv2.erode(mask,np.ones((3,3),np.uint8));ys,xs=np.where(mask>0)
    if len(xs)<12:return reject('eroded_target_too_small')
    def world_pixels(x,y):
        z=frame.depth_m[y,x];valid=np.isfinite(z)
        p=np.column_stack([(x[valid]-frame.K[0,2])*z[valid]/frame.K[0,0],(y[valid]-frame.K[1,2])*z[valid]/frame.K[1,1],z[valid]])
        return p@T_WC[:3,:3].T+T_WC[:3,3]
    points=world_pixels(xs,ys)
    if len(points)<12:return reject('insufficient_valid_target_depth')
    median=np.median(points,axis=0)
    x0,y0,w,h,_=stats[label];yy,xx=np.mgrid[max(0,y0-25):min(frame.rgb.shape[0],y0+h+25):2,max(0,x0-25):min(frame.rgb.shape[1],x0+w+25):2]
    outer=red[yy,xx]==0;px=xx[outer];py=yy[outer]
    valid_neighbors=(px+1<frame.rgb.shape[1])&(py+1<frame.rgb.shape[0]);px=px[valid_neighbors];py=py[valid_neighbors]
    def world_unfiltered(x,y):
        z=frame.depth_m[y,x]
        p=np.column_stack([(x-frame.K[0,2])*z/frame.K[0,0],(y-frame.K[1,2])*z/frame.K[1,1],z])
        return p@T_WC[:3,:3].T+T_WC[:3,3]
    center=world_unfiltered(px,py);dx=world_unfiltered(px+1,py)-center;dy=world_unfiltered(px,py+1)-center
    normal=np.cross(dx,dy);norm=np.linalg.norm(normal,axis=1)
    horizontal=np.isfinite(center).all(axis=1)&(norm>1e-10)&(np.abs(normal[:,2])>.90*norm)
    surround=center[horizontal];normals=normal[horizontal]/norm[horizontal,None]
    normals[normals[:,2]<0]*=-1
    if len(surround)<20:return reject('no_horizontal_support')
    axis=np.median(normals,axis=0);axis/=np.linalg.norm(axis)
    # Fit in the measured support-plane basis, not an assumed perfect world Z.
    # This keeps camera-relative geometry invariant to navigation pose error.
    projection=surround@axis;target_height=float(median@axis)
    good=(projection<target_height-.004)&(projection>target_height-.08)&(np.linalg.norm(surround-median,axis=1)<.13)
    values=projection[good]
    if len(values)<15:return reject('no_support_below_target')
    bins=np.round(values/.004).astype(int);ids,count=np.unique(bins,return_counts=True);chosen=ids[np.argmax(count)]
    plane=float(np.median(values[bins==chosen]));heights=points@axis
    side=points[(heights>plane+.004)&(heights<plane+height-.004)]
    if len(side)<12:return reject('insufficient_cylinder_side')
    basis1=np.cross(axis,[0.,1.,0.]);basis1/=np.linalg.norm(basis1);basis2=np.cross(axis,basis1)
    basis=np.column_stack([basis1,basis2]);side2=side@basis
    bearing=median-T_WC[:3,3];bearing/=max(np.linalg.norm(bearing),1e-8)
    initial=median@basis+radius*(bearing@basis)
    fit=least_squares(lambda c:np.linalg.norm(side2-c,axis=1)-radius,initial,loss='soft_l1',f_scale=.002,max_nfev=50)
    error=float(np.sqrt(np.mean(fit.fun**2)))
    if error>.0015 or np.linalg.norm(fit.x-initial)>.03:return reject('circle_fit_or_center_bounds')
    LAST_TARGET_DIAGNOSTIC={'reason':'detected','side_pixels':len(side),'circle_error_m':error,'support_offset':plane}
    center_world=basis@fit.x+axis*(plane+height/2)
    return dict(center_W=center_world,circle_residual_m=error,support_normal_W=axis,support_plane_offset=plane,pixels=len(points))
