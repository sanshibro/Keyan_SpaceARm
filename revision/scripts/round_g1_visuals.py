"""Scientific evidence plate and time-synchronized world-fixed footage from the preregistered seed."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import mujoco
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from round_f_plant import CHANNELS
from round_f_rotor_replay import split_model, both, font, TURNS_PER_SIM_SECOND
from round_g1_model import DATA, P, TARGET, make_plant

SEED=P['visual_seed']
KEYS=['straight','fixed','variable_nominal','variable_timed']
COLORS=dict(zip(KEYS,['#5c6670','#957348','#47779a','#247a6b']))
NAMES=dict(zip(KEYS,['Straight','Fixed bend','Variable, nominal timing','Variable, anticipatory timing']))
REVIEW=DATA/'review'; INK='#243c45'; GRAY='#536970'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.edgecolor':'#9aa7a6','axes.linewidth':.6,'xtick.color':GRAY,'ytick.color':GRAY,
                     'axes.labelcolor':INK,'text.color':INK,'figure.facecolor':'white','savefig.facecolor':'white'})


class Camera:
    def __init__(self,m,axes):
        self.m=m;self.d=mujoco.MjData(m);self.axes=axes
        self.base=m.body('uav').id;self.ids={m.geom('f_blades_%d'%b).id:k for k,b in enumerate(CHANNELS)}
        self.ren=mujoco.Renderer(m,height=480,width=480)
        self.opt=mujoco.MjvOption();self.opt.geomgroup[:]=[0,1,1,0,1,0];self.opt.sitegroup[:]=0
        self.cam=mujoco.MjvCamera();self.cam.orthographic=1;self.cam.distance=1.8
        self.cam.azimuth=98.;self.cam.elevation=-7.
        # A section view removes only the near/far jambs from this render copy.
        for name in ['jamb_negative','jamb_positive']:
            m.geom_group[m.geom('G1_'+name).id]=5
        for name in ['sill','lintel']:
            gid=m.geom('G1_'+name).id
            # Cut only the render copy to a 12 cm central strip. A full-depth
            # jamb projected obliquely would hide the opening and both witnesses.
            m.geom_size[gid,1]=.06
            m.geom_rgba[gid]=[.77,.82,.83,1.]

    def project(self,p):
        l,r=self.ren.scene.camera;eye=(np.asarray(l.pos)+r.pos)/2
        z=np.asarray(p)-eye;h=l.frustum_top-l.frustum_bottom
        return np.array([240+480*(z@np.cross(l.forward,l.up))/h,240-480*(z@l.up)/h])

    def image(self, q, phase, center, span):
        m,d=self.m,self.d
        d.qpos[:]=q;mujoco.mj_kinematics(m,d)
        rot=d.xmat[self.base].reshape(3,3)
        centers=d.xpos[self.base]+self.axes[CHANNELS]@rot.T
        self.cam.lookat[:]=center;m.vis.global_.fovy=span
        self.ren.update_scene(d,camera=self.cam,scene_option=self.opt)
        for g in self.ren.scene.geoms[:self.ren.scene.ngeom]:
            if g.objtype!=int(mujoco.mjtObj.mjOBJ_GEOM) or g.objid not in self.ids:continue
            j=self.ids[g.objid];spin=rot@both(phase[j])@rot.T
            g.pos[:]=centers[j]+spin@(g.pos.astype(float)-centers[j])
            g.mat[:]=spin@g.mat.astype(float)
        return Image.fromarray(self.ren.render().copy()).convert('RGB')


class Flight:
    def __init__(self,kind):
        case='%d-%s'%(SEED,kind);folder=DATA/'runs'/case
        self.a=dict(np.load(str(folder/'physics.npz')));w=pd.read_csv(folder/'hil.csv')
        self.r=json.loads((folder/'run.json').read_text())['release_us'];self.t=(self.a['us']-self.r)/1e6
        self.wt=(w.us.values-self.r)/1e6
        self.bend=w.net_deg.values
        motor=w[['m1','m2','m3','m4']].values
        self.phase=np.vstack((np.zeros(4),np.cumsum(np.sqrt(motor[:-1])*[1,1,-1,-1]*(np.diff(self.wt)*2*np.pi*TURNS_PER_SIM_SECOND)[:,None],axis=0)))
        self.tr=pd.read_csv(DATA/'traces'/(case+'.csv'))

    def frame(self,t,camera,center,span):
        k=int(np.argmin(abs(self.t-t)));j=int(np.argmin(abs(self.wt-t)))
        return camera.image(self.a['qpos'][k],self.phase[j],center,span),k,j


def render_plate(c,fl,exact):
    cases=[('straight','body','Body to lintel'),('variable_nominal','arm_tool','Finger to sill'),
           ('variable_timed','arm_tool','Finger to sill')]
    fig=plt.figure(figsize=(10.65,4.58))
    for k,(name,part,what) in enumerate(cases):
        e=exact['%d-%s'%(SEED,name)][part];f=fl[name]
        center=np.array([1.0,0.,2.04]);im,ii,j=f.frame(e['t'],c,center,1.06)
        assert ii==e['k']
        a,b=c.project(e['robot']),c.project(e['wall'])
        ax=fig.add_axes([.033+.327*k,.17,.295,.68]);ax.imshow(im)
        ax.plot([a[0],b[0]],[a[1],b[1]],color='#b65332',lw=1.6)
        ax.scatter([a[0],b[0]],[a[1],b[1]],s=11,facecolor='white',edgecolor='#b65332',linewidth=1.2,zorder=5)
        for z in [P['standard_gate']['sill_z'],P['standard_gate']['lintel_z']]:
            u=c.project([P['gate_center_x'],0.,z])
            ax.plot([42,441],[u[1],u[1]],color='#abbaBA',lw=.45,ls=(0,(3,4)),alpha=.7,zorder=.5)
        ax.set_xlim(0,480);ax.set_ylim(480,0);ax.axis('off')
        fig.text(.04+.327*k,.92,chr(97+k),fontweight='bold',fontsize=12)
        fig.text(.062+.327*k,.92,NAMES[name],fontsize=10.3)
        fig.text(.04+.327*k,.118,what+'   %.2f cm'%(100*e['upper_m']),color='#913e26',fontsize=11)
        fig.text(.04+.327*k,.066,'t = %.2f s     measured bend %.1f°'%(e['t'],f.bend[j]),fontsize=9,color=GRAY)
    fig.savefig(REVIEW/'closest.png',dpi=205);plt.close(fig)


def curves(fl,exact):
    fig,axes=plt.subplots(2,2,figsize=(10.6,6.05),gridspec_kw={'height_ratios':[1,1]})
    fig.subplots_adjust(left=.078,right=.975,top=.914,bottom=.19,wspace=.28,hspace=.55)
    for name,title,ax,col,thr,lim in [
            ('a','Rigid-body clearance',axes[0,0],'body_m',10,(0,49)),
            ('b','Arm and tool clearance',axes[0,1],'arm_tool_m',7,(0,49)),
            ('c','Executed arm bend',axes[1,0],'bend_deg',None,(0,83))]:
        for key in KEYS:
            t=fl[key].tr;mask=(t.t>=21)&(t.t<=36);scale=1 if col=='bend_deg' else 100
            ax.plot(t.t[mask].to_numpy(),scale*t[col][mask].to_numpy(),lw=1.35,color=COLORS[key],ls={'straight':'--','fixed':'-.','variable_nominal':'-','variable_timed':'-'}[key],clip_on=True)
        if thr:
            ax.axhline(thr,color='#a6563e',lw=.8,ls=(0,(2,2)))
            ax.text(35.85,thr-2.,'%g cm'%thr,ha='right',va='top',fontsize=8,color='#924831')
        ax.set_xlim(21,36);ax.set_ylim(*lim);ax.set_xticks([22,25,28,31,34])
        ax.set_ylabel('Angle (°)' if col=='bend_deg' else 'Lower bound (cm)')
        ax.set_xlabel('Flight time (s)')
        ax.set_title(name+'   '+title,loc='left',fontsize=10.4,fontweight='bold')
        ax.grid(axis='y',color='#e5e9e8',lw=.55)
    ax=axes[1,1];ax.set_title('d   Paired arm and tool minima',loc='left',fontsize=10.4,fontweight='bold')
    for s in P['seed_order']:
        yy=[100*exact[s+'-'+v]['arm_tool']['upper_m'] for v in KEYS[-2:]]
        ax.plot([0,1],yy,color='#a0aaa9',lw=.8)
        ax.scatter([0,1],yy,c=[COLORS[k] for k in KEYS[-2:]],s=18,zorder=4,edgecolor='white',linewidth=.5)
    ax.set_xlim(-.20,1.20);ax.set_ylim(9.5,17.1);ax.set_xticks([0,1]);ax.set_xticklabels(['Nominal','Anticipatory'])
    ax.set_ylabel('Convex-surface distance (cm)');ax.grid(axis='y',color='#e5e9e8',lw=.55)
    handles=[plt.Line2D([0],[0],color=COLORS[k],lw=2,linestyle='--' if k=='straight' else '-.' if k=='fixed' else '-') for k in KEYS]
    fig.legend(handles,[NAMES[k] for k in KEYS],loc='lower center',bbox_to_anchor=(.51,.02),ncol=4,frameon=False,fontsize=8.7)
    fig.savefig(REVIEW/'curves.png',dpi=205);plt.close(fig)


def video(cam,fl):
    W,H=1248,625;fps=16
    paths=[('variable_nominal','常规时序'),('variable_timed','提前弯臂、延后收臂')]
    dest=REVIEW/'comparison.mp4'
    cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}',
         '-r',str(fps),'-i','-','-an','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-preset','fast',
         '-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    fonts=[font(25,True),font(17),font(19,True),font(16)]
    stamps=np.arange(14.,46.001,1/fps)
    try:
        for it,t in enumerate(stamps):
            canvas=Image.new('RGB',(W,H),'white');pen=ImageDraw.Draw(canvas)
            pen.text((28,12),'单门穿越与门后夹爪到位',font=fonts[0],fill=INK)
            pen.text((W-245,18),'飞行时间  %05.2f s'%t,font=fonts[2],fill=INK)
            pen.line((28,56,W-28,56),fill='#bbc8c8')
            for j,(mode,title) in enumerate(paths):
                x=31+615*j;f=fl[mode]
                im,kk,wi=f.frame(t,cam,[1.,0.,1.98],2.42)
                # Use the middle 64% of the square image vertically for the landscape frame.
                full=im.resize((582,582),Image.Resampling.LANCZOS if hasattr(Image,'Resampling') else Image.LANCZOS)
                im=full.crop((0,91,582,477))
                ink=ImageDraw.Draw(im)
                def project(z):
                    xy=cam.project(z)*582/480
                    return xy[0],xy[1]-91
                path=f.a['tcp'][max(0,kk-125):kk+1:3]
                if len(path)>1:ink.line([project(z) for z in path],fill='#2e8375',width=2)
                a,b=project(TARGET)
                for dx,dy,ex,ey in [(-8,0,-3,0),(3,0,8,0),(0,-8,0,-3),(0,3,0,8)]:
                    ink.line((a+dx,b+dy,a+ex,b+ey),fill='#a14d32',width=2)
                canvas.paste(im,(x,106));pen.text((x,67),title,font=fonts[2],fill=COLORS[mode])
                k=int(np.argmin(abs(f.tr.t.values-t)));row=f.tr.iloc[k]
                pen.text((x+4,498),'臂体与夹爪净空下界  %.1f cm     实际弯角  %.1f°'%(100*row.arm_tool_m,row.bend_deg),font=fonts[1],fill=INK)
                pen.text((x+4,531),'夹爪至共同目标  %.1f cm'%(100*row.target_m),font=fonts[3],fill=GRAY)
            pen.line((28,572,W-28,572),fill='#c5cece')
            pen.text((29,582),'同一世界机位与飞行时钟   ·   绿色轨迹为夹爪末端   ·   十字为目标   ·   侧视剖面省略左右门柱   ·   桨叶按推力降速显示',font=fonts[3],fill=GRAY)
            if abs(t-28.)<1e-7:canvas.save(REVIEW/'matched.png')
            proc.stdin.write(canvas.tobytes())
    finally:
        proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('ffmpeg encoding failed')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--video',action='store_true');a=ap.parse_args()
    REVIEW.mkdir(exist_ok=True);fl={k:Flight(k) for k in KEYS}
    exact=json.loads((DATA/'convex-distance.json').read_text())['runs']
    with tempfile.TemporaryDirectory() as tmp:
        m,ax,_=split_model(make_plant(),Path(tmp));cam=Camera(m,ax)
        try:
            render_plate(cam,fl,exact);curves(fl,exact)
            if a.video:video(cam,fl)
        finally:cam.ren.close()


if __name__=='__main__':main()
