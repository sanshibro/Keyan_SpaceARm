"""Frame compensation shared by future RGB-D estimator. Inputs are estimates."""
import numpy as np
from qingyu_model import ARC_LENGTH, BASE_LENGTH

def relative_camera(T_WU,T_UB,T_WC,stamp_u,stamp_c,max_skew=.01):
    if abs(stamp_u-stamp_c)>max_skew:
        raise ValueError('Unsynchronised navigation and camera estimates')
    for T in (T_WU,T_UB,T_WC):
        if np.shape(T)!=(4,4) or not np.isfinite(T).all():raise ValueError('Invalid pose')
    return np.linalg.solve(T_WU@T_UB,T_WC)

def orientation_shape(T_BC,T_TC):
    """Orientation-only PCC inversion; position residual must be audited separately."""
    R=(T_BC@np.linalg.inv(T_TC))[:3,:3]
    tangent=R[:,2];theta=np.arctan2(np.linalg.norm(tangent[:2]),tangent[2])
    if theta<1e-6:return np.zeros(2)
    return theta*tangent[:2]/np.linalg.norm(tangent[:2])

def pcc_tool_transform(bend,length=ARC_LENGTH,base_length=BASE_LENGTH):
    """Forward PCC pose from a measured bend vector, in the mount frame."""
    bend=np.asarray(bend,dtype=float).reshape(2)
    theta=float(np.linalg.norm(bend))
    phi=float(np.arctan2(bend[1],bend[0]))
    BT=np.eye(4)
    if theta<1e-8:
        BT[:3,3]=[0,0,base_length+length]
    else:
        axis=np.array([-np.sin(phi),np.cos(phi),0.]);x,y,z=axis
        K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        BT[:3,:3]=np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*(K@K)
        radial=length*(1-np.cos(theta))/theta
        BT[:3,3]=[radial*np.cos(phi),radial*np.sin(phi),base_length+length*np.sin(theta)/theta]
    return BT

def pcc_camera_pose(T_WU,T_UB,T_WC_visual,T_TC,length=ARC_LENGTH,base_length=BASE_LENGTH):
    """Project visual orientation onto the calibrated single-segment PCC prior.

    Position is model-constrained, not independent camera-translation odometry.
    Use only for low-contact survey; residuals must remain separately reported.
    """
    relative=relative_camera(T_WU,T_UB,T_WC_visual,0.,0.)
    b=orientation_shape(relative,T_TC)
    return T_WU@T_UB@pcc_tool_transform(b,length,base_length)@T_TC,b
