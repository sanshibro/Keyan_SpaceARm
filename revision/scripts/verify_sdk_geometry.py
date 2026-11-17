"""Compare compiled gripper visuals against the unmodified installed SDK STL."""
from pathlib import Path
import sys,struct,json,hashlib
import numpy as np,mujoco
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'mujoco_closed_loop'))
from super_metric_model import build_super_metric
from model import SimArmBridge
source=ROOT/'mujoco_closed_loop/vendor/qingyu_continuum_arm/meshes/gripper.STEP.stl'
raw=source.read_bytes();n=struct.unpack_from('<I',raw,80)[0]
dtype=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')])
v=np.frombuffer(raw,dtype=dtype,count=n,offset=84)['v'].reshape(-1,3).astype(float)
R=np.array([[-1,0,0],[0,0,1],[0,1,0]],dtype=float)
expected=(v*.001+np.array([.0021,.0708,-.023]))@R.T
expected=np.unique(np.round(expected,9),axis=0)
xml,info=build_super_metric(fixed=True);m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
SimArmBridge().set_kinematic_shape(m,d,np.pi/2,np.pi);mujoco.mj_forward(m,d)
tool=m.body('tool').id;Rt=d.xmat[tool].reshape(3,3);pt=d.xpos[tool];clouds=[]
for name in ['housing','finger0','finger1','finger2']:
    g=m.geom('visual_gripper_'+name).id;mid=m.geom_dataid[g]
    off=m.mesh_vertadr[mid];count=m.mesh_vertnum[mid]
    local=m.mesh_vert[off:off+count]
    world=local@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
    clouds.append((world-pt)@Rt)
actual=np.vstack(clouds)
dist1=cKDTree(expected).query(actual)[0];dist2=cKDTree(actual).query(expected)[0]
error=max(float(dist1.max()),float(dist2.max()))
assert error<2e-7,error
result=dict(platform_id=info['platform_id'],source_sha256=hashlib.sha256(raw).hexdigest(),
    sdk_original_triangles=n,reference_vertices=len(expected),compiled_vertices=len(actual),
    max_bidirectional_vertex_distance_m=error,reference_pose='zero finger slide, SDK URDF+TF transform',
    units='m',passed=True,scope='native surface and rigid installation; gripper actuation not calibrated')
out=ROOT/'revision/data/gripper_review';out.mkdir(exist_ok=True,parents=True)
(out/'native_geometry_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
