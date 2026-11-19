"""Actual surface paths, ADC samples and independent angle labels on one held-out bench motion."""
import json
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from round_shape_pixels import OUTPUT
from round_g1_model import make_plant
from round_shape_sensors import NODES,RING,IDX,FRACTION,features,predict,vector_angles
from round_wrist_visuals import INK,GREY,GREEN
def main():
    raw=np.load(str(OUTPUT/'bench/test-00.npz'));t=raw['t'];counts=raw['counts']
    model=json.loads((OUTPUT/'bench/predictor.json').read_bytes());X,tare=features(counts,t)
    th,phi=vector_angles(raw['mocap_b']);estimate=vector_angles(predict(model,X))[0]
    plant=make_plant();stage=int(np.argmax(th));plant.d.qpos[plant.arm.qpos_ids]=raw['plant_joint'][stage]
    mujoco.mj_forward(plant.m,plant.d)
    mount=plant.d.body('mount');R=mount.xmat.reshape(3,3);pos=mount.xpos
    ids=[plant.m.body(n).id for n in NODES]
    centers=plant.d.xpos[ids].copy();axes=plant.d.xmat[ids].reshape(-1,3,3)
    strips=centers[:,None,:]+RING[None,:,:]@axes.transpose(0,2,1)
    gauges=strips[IDX]*(1-FRACTION[:,None,None])+strips[IDX+1]*FRACTION[:,None,None]
    planar=np.array([np.cos(np.deg2rad(phi[stage])),np.sin(np.deg2rad(phi[stage])),0.])
    def project(p):
        local=(p-pos)@R
        return np.stack((1000*(local@planar),1000*local[...,2]),axis=-1)
    colors=['#ae6743','#427c99','#348067','#856793']
    fig=plt.figure(figsize=(10.2,4.5));ax=fig.add_axes([.068,.21,.248,.61])
    c=project(centers);ax.plot(c[:,0],c[:,1],color='#637578',lw=.9,ls='--')
    a=project(strips)
    from scipy.spatial import ConvexHull
    # Fill link-to-link quadrilaterals to retain the physical concave outline.
    for i in range(len(a)-1):
        points=np.concatenate((a[i],a[i+1]));poly=points[ConvexHull(points).vertices]
        ax.fill(poly[:,0],poly[:,1],color='#e3eae8',edgecolor='#aebbb9',lw=.55,zorder=-1)
    for j,col in enumerate(colors):
        path=project(gauges[:,j]);ax.plot(*path.T,color=col,lw=2.6,marker='.',ms=3)
    ax.invert_yaxis();ax.set_aspect('equal');ax.set_xlabel('In-plane offset (mm)');ax.set_ylabel('Arm-root axis (mm)')
    fig.text(.046,.944,'a   Virtual strip locations',fontsize=10.5,fontweight='bold')
    fig.text(.051,.090,'Four surface paths, 60 mm each',fontsize=8.7,color=GREY)
    fig.text(.051,.044,'Arm posed at %.1f°; section view'%th[stage],fontsize=8.1,color=GREY)
    p=fig.add_axes([.423,.615,.55,.26]);q=fig.add_axes([.423,.183,.55,.24],sharex=p)
    for j,col in enumerate(colors):p.plot(t,counts[:,j]-tare[j],color=col,lw=1,label='S%d'%(j+1))
    p.axhline(0,color='#a5b1b1',lw=.5);p.set_ylabel('ADC – tare (counts)');p.tick_params(labelbottom=False)
    for panel in (p,q):panel.grid(axis='y',color='#e0e7e5',lw=.5);panel.set_xlim(0,16)
    fig.text(.40,.944,'b   Four simulated strain-gauge channels',fontsize=10.5,fontweight='bold')
    p.legend(frameon=False,ncol=4,fontsize=8,loc='lower right',bbox_to_anchor=(1,1.025),columnspacing=.9)
    q.plot(t,th,color=INK,lw=2,label='Virtual mocap');q.plot(t,estimate,color=GREEN,lw=1.15,ls='--',label='From ADC')
    q.set_ylabel('Bend (deg)');q.set_xlabel('Bench time (s)');q.set_yticks([0,40,80]);q.legend(ncol=2,frameon=False,fontsize=8,loc='upper right')
    q.set_title('c   Angle on a held-out motion',loc='left',fontweight='bold',fontsize=10)
    fig.savefig(OUTPUT/'review/gauges.png',dpi=155);plt.close(fig)
if __name__=='__main__':main()
