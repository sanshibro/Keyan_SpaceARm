"""CAD self projection driven only by estimated shape. Never consumes external scene IDs."""
from xml.etree.ElementTree import fromstring,tostring
import numpy as np,mujoco,cv2
from model import model_xml,SimArmBridge
from sensors import OPTICAL_FROM_GL

class EstimatedSelfMask:
    def __init__(self,camera='wrist_rgbd',profile='grasp_visible',width=480,height=360,nav_profile='front',position_uncertainty_m=.020,super_visual=False):
        if super_visual:
            from stage1_scene import build_flight_base
            root=build_flight_base()
        else:root=fromstring(model_xml(camera_profile=profile,nav_camera_profile=nav_profile))
        world=root.find('worldbody')
        for geom in list(world.findall('geom')):world.remove(geom)
        self.model=mujoco.MjModel.from_xml_string(tostring(root,encoding='unicode'))
        self.data=mujoco.MjData(self.model);self.camera=camera;self.bridge=SimArmBridge()
        self.renderer=mujoco.Renderer(self.model,height=height,width=width);self.renderer.enable_depth_rendering()
        self.position_uncertainty_m=position_uncertainty_m
    def mask(self,frame,bending):
        theta=np.linalg.norm(bending);phi=np.arctan2(bending[1],bending[0])
        # This is an internal kinematic rendering model, not the physical simulation.
        self.bridge.set_kinematic_shape(self.model,self.data,theta,phi)
        mujoco.mj_forward(self.model,self.data);self.renderer.update_scene(self.data,camera=self.camera)
        predicted=self.renderer.render().copy()
        # The physical continuum can depart from the estimated PCC surface.
        # Discard ambiguous silhouette rays within the existing 20 mm model
        # endpoint tolerance; never mark those rays free. Keep clearly closer
        # foreground returns, so an obstacle is not erased by the self mask.
        radius=int(np.ceil(frame.K[0,0]*self.position_uncertainty_m/.15))
        kernel=np.ones((2*radius+1,2*radius+1),np.uint8)
        valid=predicted<6
        near=cv2.erode(np.where(valid,predicted,1e6).astype(np.float32),kernel)
        far=cv2.dilate(np.where(valid,predicted,0).astype(np.float32),kernel)
        mask=np.isfinite(frame.depth_m)&(far>0)&(frame.depth_m>=near-.035)&(frame.depth_m<=far+.035)
        # A moving continuum need not share the PCC renderer's visible front
        # surface. Also reject endpoints near ANY calibrated robot primitive,
        # including surfaces hidden in the predicted silhouette. No scene IDs
        # or physical joint state enter this internal kinematic model.
        yy,xx=np.where(np.isfinite(frame.depth_m)&(frame.depth_m<1.)&~mask)
        z=frame.depth_m[yy,xx]
        optical=np.column_stack([(xx-frame.K[0,2])*z/frame.K[0,0],(yy-frame.K[1,2])*z/frame.K[1,1],z])
        cid=self.model.camera(self.camera).id
        points=optical@(self.data.cam_xmat[cid].reshape(3,3)@OPTICAL_FROM_GL).T+self.data.cam_xpos[cid]
        ambiguous=np.zeros(len(points),dtype=bool)
        for i in range(self.model.ngeom):
            if self.model.geom_group[i] not in (1,2):continue
            size=self.model.geom_size[i];typ=self.model.geom_type[i]
            p=(points-self.data.geom_xpos[i])@self.data.geom_xmat[i].reshape(3,3)
            if typ in (mujoco.mjtGeom.mjGEOM_BOX,mujoco.mjtGeom.mjGEOM_MESH):
                q=abs(p)-size;distance=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(axis=1),0)
            elif typ==mujoco.mjtGeom.mjGEOM_CAPSULE:
                distance=np.sqrt(np.sum(p[:,:2]**2,axis=1)+np.maximum(abs(p[:,2])-size[1],0)**2)-size[0]
            elif typ==mujoco.mjtGeom.mjGEOM_CYLINDER:
                q=np.column_stack([np.linalg.norm(p[:,:2],axis=1)-size[0],abs(p[:,2])-size[1]])
                distance=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(axis=1),0)
            else:continue
            ambiguous|=distance<self.position_uncertainty_m
        mask[yy[ambiguous],xx[ambiguous]]=True
        return mask
    def close(self):self.renderer.close()
