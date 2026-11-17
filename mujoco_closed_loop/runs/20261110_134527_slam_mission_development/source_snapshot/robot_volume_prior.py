"""Observed self-volume cells from calibrated collision geometry and estimated configuration.

This is voxel-grid footprint clearing, not an external free-space survey. It never
clears occupied cells and never adds a safety margin outside the robot geometry.
"""
from xml.etree.ElementTree import fromstring,tostring
import numpy as np,mujoco
from model import model_xml,SimArmBridge

class RobotVolumePrior:
    def __init__(self,profile='grasp_visible',nav_profile='front',approved_stage1=False):
        if approved_stage1:
            from stage1_scene import build_collision_reference
            root=build_collision_reference()
        else:root=fromstring(model_xml(camera_profile=profile,nav_camera_profile=nav_profile))
        root.find("worldbody/body[@name='uav']").set('pos','0 0 0')
        world=root.find('worldbody')
        for g in list(world.findall('geom')):world.remove(g)
        self.model=mujoco.MjModel.from_xml_string(tostring(root,encoding='unicode'));self.data=mujoco.MjData(self.model);self.bridge=SimArmBridge();self.cache={}
    def integrate(self,mapper,T_WU,bending):
        theta=np.linalg.norm(bending);phi=np.arctan2(bending[1],bending[0]);self.bridge.set_kinematic_shape(self.model,self.data,theta,phi);mujoco.mj_forward(self.model,self.data)
        all_indices=[]
        for i in range(self.model.ngeom):
            if not self.model.geom_contype[i]:continue
            typ=self.model.geom_type[i];size=self.model.geom_size[i].copy();key=(i,mapper.resolution)
            if key not in self.cache:
                radius=max(.001,size[0]-(.015 if typ==mujoco.mjtGeom.mjGEOM_CAPSULE else .005))
                extent=size.copy()
                if typ==mujoco.mjtGeom.mjGEOM_CAPSULE:extent=np.array([radius,radius,size[1]+radius])
                elif typ==mujoco.mjtGeom.mjGEOM_CYLINDER:extent=np.array([radius,radius,size[1]])
                elif typ==mujoco.mjtGeom.mjGEOM_BOX:extent=np.maximum(.001,size-.005)
                else:continue
                axes=[np.linspace(-v,v,max(2,int(np.ceil(2*v/(mapper.resolution/3)))+1)) for v in extent]
                xyz=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
                if typ==mujoco.mjtGeom.mjGEOM_CAPSULE:xyz=xyz[np.sum(xyz[:,:2]**2,axis=1)+np.maximum(abs(xyz[:,2])-size[1],0)**2<=radius**2+1e-12]
                elif typ==mujoco.mjtGeom.mjGEOM_CYLINDER:xyz=xyz[np.sum(xyz[:,:2]**2,axis=1)<=radius**2+1e-12]
                self.cache[key]=xyz
            local=self.cache[key]@self.data.geom_xmat[i].reshape(3,3).T+self.data.geom_xpos[i]
            points=local@T_WU[:3,:3].T+T_WU[:3,3];all_indices.extend(map(tuple,np.floor(points/mapper.resolution+1e-9).astype(int)))
        cells=set(all_indices);conflicts=cells&mapper.occupied;mapper.observed.update(cells-conflicts)
        return {'self_volume_voxels':len(cells),'occupied_conflicts_preserved':len(conflicts),'geometry_erosion_m':.005,'added_external_margin_m':0.}
