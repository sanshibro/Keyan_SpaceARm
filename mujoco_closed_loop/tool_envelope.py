"""Sample explicit rigid camera/gripper/payload collision envelopes for V2 adaptation."""
from xml.etree.ElementTree import fromstring
import numpy as np,mujoco
from model import model_xml

def collision_samples(camera_profile='grasp_visible',loaded=False,spacing=.012,approved_stage1=False):
    if approved_stage1:
        from stage1_scene import build_collision_reference
        root=build_collision_reference()
    else:root=fromstring(model_xml(camera_profile=camera_profile))
    m=mujoco.MjModel.from_xml_string(__import__('xml.etree.ElementTree',fromlist=['tostring']).tostring(root,encoding='unicode'));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    tool=m.body('tool').id;R=d.xmat[tool].reshape(3,3);origin=d.xpos[tool];points=[]
    for i in range(m.ngeom):
        body=m.geom_bodyid[i]
        if body!=tool and m.body_parentid[body]!=tool:continue
        if not m.geom_contype[i]:continue
        size=m.geom_size[i];typ=m.geom_type[i]
        # Match the contact-free CAD volume used by RobotVolumePrior.  The
        # planner adds the arm radius and clearance outside these samples;
        # surface samples one voxel beyond that volume become false unknowns.
        if typ==mujoco.mjtGeom.mjGEOM_BOX:ext=np.maximum(.001,size-.005)
        elif typ==mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius=max(.001,size[0]-.015)
            ext=np.r_[radius,radius,size[1]+radius]
        else:continue
        axes=[np.linspace(-x,x,max(2,int(np.ceil(2*x/spacing))+1)) for x in ext]
        grid=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
        world=grid@d.geom_xmat[i].reshape(3,3).T+d.geom_xpos[i]
        points.extend((world-origin)@R)
    if loaded:
        # Conservative full bounding volume of the controlled cylinder;
        # four diagonal corners alone leave entire faces unsampled.
        axes=[np.linspace(-.012,.012,max(3,int(np.ceil(.024/spacing))+1)),
              np.linspace(-.012,.012,max(3,int(np.ceil(.024/spacing))+1)),
              np.linspace(.108,.148,max(3,int(np.ceil(.040/spacing))+1))]
        points.extend(np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3))
    return tuple(map(tuple,np.unique(np.round(points,9),axis=0)))
