"""Rigid hardware mass moment relative to the reused V2 mass approximation."""
import numpy as np

def rigid_com_correction(bending,total_mass,survey=False,loaded=False,approved_stage1=False):
    theta=float(np.linalg.norm(bending));phi=float(np.arctan2(bending[1],bending[0]))
    axis=np.array([-np.sin(phi),np.cos(phi),0.]);x,y,z=axis
    K=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    rotation=np.diag([-1.,1.,-1.])@(np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*(K@K))
    if approved_stage1:
        moment=(.02*np.array([0,.030,-.002])+.13*np.array([0,.080,.010])
                +.05*np.array([0,0,.112]))
    else:
        moment=(.065*np.array([0,0,.045])+.015*np.array([.0351335,.0771335,.0054115])
                +.07*np.array([.070267,.129267,.010823])+.05*np.array([0,0,.112]))
    residual=rotation@(moment-np.array([0,0,.20*.08]))
    if loaded:residual+=rotation@np.array([0,0,.04*(.128-.08)])
    if approved_stage1:
        residual+=.08*np.array([0,.105,.015])
    elif survey:
        # These 80 g were subtracted from the fuselage's origin mass.
        residual+=.072*np.array([.166,0,-.13])+.008*np.array([.153,0,-.085])
    return residual/total_mass
