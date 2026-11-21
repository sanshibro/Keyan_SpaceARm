"""Select recorded RGB-D exposures by visible native geometry at each inspection feature."""
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw
from m1_layout import SEEDS
from m1_replay import build,option,writej
from m1_media import Flight,title
from m1_plates import DATA,FILM
def main():
    p=build();f=Flight(DATA/('coupled-'+str(SEEDS[0])),p.scene_digest)
    rr=mujoco.Renderer(p.m,height=180,width=240);rr.enable_segmentation_rendering();opt=option()
    registry=[p.m.geom(i).name for i in range(p.m.ngeom)]
    groups=[('Service opening',('M1_sill','M1_lintel','M1_jamb'),.16,.98,.29),
            ('Asymmetric pipe',('M1_asymmetric_riser','M1_instrument','M1_riser_return'),.78,1.70,.16),
            ('Inspection flange',('M1_flange_front','M1_flange_lip'),3.19,3.34,.35)]
    image=Image.new('RGB',(1518,434),'white');dr=ImageDraw.Draw(image);states=[]
    try:
        for col,(name,pref,xmin,xmax,target) in enumerate(groups):
            ids=np.flatnonzero([s.startswith(pref) for s in registry]);best=None
            for oid in range(0,len(f.index),8):
                o,j=f.nearest(f.index[oid]['t'])
                if not xmin<=f.q[j,0]<=xmax:continue
                if col==2 and abs(o['t']-(f.metrics.get('first_pass_s') or f.opt_t[-1]))>1:continue
                d=p.d;d.qpos[:]=f.q[j];d.eq_active[0]=0;mujoco.mj_kinematics(p.m,d);mujoco.mj_camlight(p.m,d)
                rr.update_scene(d,camera='wrist_rgbd',scene_option=opt);seg=rr.render()
                mask=(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&np.isin(seg[:,:,0],ids)
                frac=mask.mean()
                if frac<.012:continue
                yy,xx=np.nonzero(mask);cen=np.array([xx.mean()/240,yy.mean()/180])
                cost=abs(frac-target)+.22*np.linalg.norm(cen-[.5,.53])
                if best is None or cost<best[0]:best=(cost,o,frac,cen)
            if best is None:raise RuntimeError('No registered exposure of '+name)
            _,o,frac,cen=best;src=f.wrist(o);px=14+505*col
            image.paste(src,(px,41));title(dr,(px,5),name,22,True,True)
            title(dr,(px,409),'Exposure %.2f s   /   delivery %.2f s'%(o['t'],o['arrival_t']),16)
            src.save(FILM/('wrist-station-%d.png'%col))
            states.append(dict(feature=name,exposure=o['t'],delivery=o['arrival_t'],frame=o['frame'],
                       native_rgb_sha256=o['rgb_sha256'],qpos_sha256=o['sensor_qpos_sha256'],visible_fraction_240x180=frac,feature_centre_normalized=cen))
        image.save(FILM/'wrist-stations.png');writej(FILM/'wrist-stations.json',states)
    finally:rr.close()
if __name__=='__main__':main()
