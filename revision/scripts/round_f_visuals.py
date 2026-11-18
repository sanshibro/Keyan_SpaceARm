"""Standalone candidate pages and full-duration replay of the logged F flight."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import mujoco
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image,ImageDraw,ImageFont
from round_f_plant import Plant,LAUNCH

INK='#243740';MUTE='#465b67';NAVY='#315e78';TEAL='#19746c';OCHRE='#a85d33';GRID='#d6dfe1';WASH='#edf4f1'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8.7,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#73818a','axes.labelcolor':INK,'text.color':INK,'axes.linewidth':.7,'xtick.color':INK,'ytick.color':INK})


def write_media(folder,out):
    plant=Plant();m,d=plant.m,plant.d
    p=folder/'runs/final'
    z=np.load(str(p/'physics.npz'));run=json.loads((p/'run.json').read_text())
    w=pd.read_csv(folder/'display_50hz.csv')
    tt=(z['us']-run['release_us'])/1e6
    t=w.since_release_s.values
    render=mujoco.Renderer(m,height=460,width=460)
    m.vis.global_.fovy=.82
    cam=mujoco.MjvCamera();cam.orthographic=1;cam.distance=1.05;cam.azimuth=113;cam.elevation=-10
    opts=mujoco.MjvOption();opts.geomgroup[0]=0;opts.geomgroup[3]=0;opts.sitegroup[:]=0
    def state(stamp):
        k=int(np.argmin(abs(tt-stamp)));d.qpos[:]=z['qpos'][k];mujoco.mj_forward(m,d)
        cam.lookat[:]=d.qpos[:3]+[.008,0,-.17]
        render.update_scene(d,camera=cam,scene_option=opts)
        return Image.fromarray(render.render().copy()).convert('RGB')
    for k,stamp in enumerate([12.,17.,30.]):state(stamp).save(out/('pose%d.png'%k))
    fnt='/usr/share/fonts/truetype/dejavu/'
    big=ImageFont.truetype(fnt+'DejaVuSerif-Bold.ttf',23);small=ImageFont.truetype(fnt+'DejaVuSans.ttf',15)
    med=ImageFont.truetype(fnt+'DejaVuSans.ttf',18)
    pending=out/'replay.part.mp4'
    ff=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x554','-r','20','-i','-',
                         '-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(pending)],stdin=subprocess.PIPE)
    def xy(e,h):return (round(505+(e+.08)*429),round(416-h*353))
    def text_at(draw,pos,txt,font=small,fill=INK):draw.text(pos,txt,font=font,fill=fill)
    for j in range(391):
        stamp=j/10;i=int(np.argmin(abs(t-stamp)));a=w.iloc[i]
        canvas=Image.new('RGB',(960,554),'white');draw=ImageDraw.Draw(canvas)
        text_at(draw,(22,16),'SUPER flight with a moving continuum arm',big)
        text_at(draw,(707,20),'PX4 SITL   /   MuJoCo',med)
        draw.line((22,52,936,52),fill=GRID)
        canvas.paste(state(stamp),(14,64))
        draw=ImageDraw.Draw(canvas)
        text_at(draw,(494,72),'Motion in the world frame',med)
        for h in [0,.2,.4,.6,.8]:
            x,y=xy(-.04,h);draw.line((x,y,923,y),fill=GRID);text_at(draw,(482,y-9),'%.1f'%h)
        for e in [0,.3,.6,.9]:
            x,y=xy(e,0);draw.line((x,y,x,y+4),fill=MUTE);text_at(draw,(x-10,y+6),'%.1f'%e)
        text_at(draw,(670,444),'East (m)')
        text_at(draw,(496,95),'Height above launch (m)')
        ref=w[['rx','rz']].values;pos=w[['ex','ez']].values
        draw.line([xy(*v) for v in ref[::20]],fill='#b7bcbe',width=2)
        line=[xy(*v) for v in pos[:i+1:3]]
        if len(line)>1:draw.line(line,fill=TEAL,width=3)
        x,y=xy(a.rx,a.rz);draw.rectangle((x-5,y-5,x+5,y+5),outline=OCHRE,width=2)
        x,y=xy(a.ex,a.ez);draw.ellipse((x-4,y-4,x+4,y+4),fill=TEAL)
        text_at(draw,(496,478),'Square   request',fill=OCHRE);text_at(draw,(698,478),'Dot   vehicle origin',fill=TEAL)
        draw.line((20,506,940,506),fill=GRID)
        text_at(draw,(22,520),'%05.1f s    %s'%(stamp,a.phase.upper()),med)
        text_at(draw,(312,520),'Arm   %.1f° issued   /   %.1f° integrated'%(a.issued_deg,a.net_deg),med)
        text_at(draw,(760,522),'Replay speed   2×',small)
        if j==300:canvas.save(out/'poster.png')
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    if ff.wait():raise RuntimeError('Replay encoder error')
    pending.replace(out/'replay.mp4')
    render.close()


def axis(fig,box,head):
    a=fig.add_axes(box);a.grid(axis='y',color=GRID,linewidth=.55);a.set_axisbelow(True)
    a.set_title(head,loc='left',fontweight='bold',fontsize=10,pad=6)
    return a

def times(a):
    a.axvspan(22,36,color=WASH,zorder=-5,lw=0)
    for k in (9,14,22,36):a.axvline(k,color=GRID,lw=.6)
    a.set_xlim(0,39);a.set_xticks([0,9,14,22,30,36,39])

def intro(fig,page,title,subtitle):
    fig.text(.072,.955,'FLIGHT DYNAMICS     /     SOFTWARE-IN-THE-LOOP',fontsize=8.3,color=MUTE)
    fig.text(.928,.955,'%d / 2'%page,ha='right',fontsize=8.3,color=MUTE)
    fig.text(.072,.895,title,size=19,weight='bold')
    fig.text(.073,.855,subtitle,fontsize=9.6)
    fig.add_artist(plt.Line2D([.072,.928],[.83,.83],transform=fig.transFigure,color=GRID,lw=.7))

def figures(raw,out):
    s=json.loads((raw/'summary.json').read_text());case=s['cases'][-1];rec=case['reconstruction'];g=pd.read_csv(raw/'display_50hz.csv')
    t=g.since_release_s.values;poses=[(12,'Station keeping'),(17,'Eastward translation'),(30,'Arm hold')]
    rr=pd.read_csv(raw/'runs/reference/hil.csv.gz');rr=rr[rr.phase=='bend']
    rf=np.linalg.norm(rr[['ex','ey','ez']].values-rr[['rx','ry','rz']].values,axis=1)*100
    b=g[g.phase=='bend'];be=np.linalg.norm(b[['ex','ey','ez']].values-b[['rx','ry','rz']].values,axis=1)*100
    def savecropped(fig,n,box):
        from matplotlib.transforms import Bbox
        fig.savefig(out/n,dpi=190,bbox_inches=Bbox.from_extents(*box).transformed(fig.transFigure).transformed(fig.dpi_scale_trans.inverted()),pad_inches=0)
    with PdfPages(str(out/'review.pdf')) as pdf:
        f=plt.figure(figsize=(8.3,9.2));intro(f,1,'Free flight with a moving arm','Nominal unloaded SUPER assembly  ·  PX4 v1.13.3  ·  native EKF2  ·  MuJoCo 3.2.3')
        for k,(at,title) in enumerate(poses):
            x=.078+.286*k;a=f.add_axes([x,.541,.267,.240])
            a.imshow(plt.imread(str(out/('pose%d.png'%k))));a.axis('off')
            i=int(np.argmin(abs(t-at)));f.text(x,.794,chr(97+k)+'   '+title,fontsize=10.2,weight='bold')
            f.text(x,.527,'%.0f s   /   E %.2f m   /   %.1f°'%(at,g.ex.iloc[i],g.net_deg.iloc[i]),fontsize=8.5,color=MUTE)
        savecropped(f,'cad.png',[.053,.508,.947,.823])
        a=axis(f,[.13,.321,.79,.139],'d   Translation and altitude')
        times(a);a.plot(t,g.ex.values,c=NAVY,lw=1.8,label='East');a.plot(t,g.ez.values,c=TEAL,lw=1.8,label='Height')
        a.plot(t,g.rx.values,'--',c=NAVY,lw=.9,label='East request');a.plot(t,g.rz.values,':',c=MUTE,lw=1.4,label='Height request')
        a.set_ylabel('Position (m)');a.legend(ncol=4,fontsize=7.5,loc='lower right',frameon=False);a.tick_params(labelbottom=False)
        a=axis(f,[.13,.146,.79,.117],'e   Arm response during position hold')
        times(a);a.plot(t,g.issued_deg.values,'--',c=OCHRE,lw=1.2,label='Issued');a.plot(t,g.net_deg.values,c=TEAL,lw=1.7,label='Integrated')
        a.set(ylabel='Bend (deg)',xlabel='Time after release of launch support (s)',ylim=(-4,128),yticks=[0,60,120])
        a.legend(loc='upper left',frameon=False,ncol=2,fontsize=8)
        f.text(.073,.065,'All three poses use the integrated joint state and the same camera angle. The 14 s shaded interval',fontsize=8.6)
        f.text(.073,.043,'holds the vehicle reference fixed while the arm bends and returns. Position is relative to launch.',fontsize=8.6)
        pdf.savefig(f);plt.close(f)
        f=plt.figure(figsize=(8.3,9.2));intro(f,2,'Position error during arm motion','One flight-control parameter changes across paired runs. The arm command and task geometry are fixed.')
        a=axis(f,[.12,.610,.80,.171],'a   Three-dimensional position error over one bend and return')
        a.plot((rr.us.values-rr.us.values[0])/1e6,rf,c=OCHRE,lw=1.2,label='Original pitch-rate integral gain, 0.20')
        a.plot(b.since_release_s.values-22,be,c=TEAL,lw=1.6,label='Revised gain, 0.40');a.set(xlim=(0,14),ylabel='Error (cm)',xlabel='Time since arm command began (s)')
        a.legend(loc='upper left',frameon=False,fontsize=7.6)
        a=axis(f,[.12,.345,.40,.16],'b   Bend-hold RMS error')
        xs=[[c['phases']['bend']['rms_3d_m']*100 for c in s['cases'] if np.isclose(c['pitch_rate_i'],i)] for i in [.2,.4]]
        assert [len(v) for v in xs]==[2,3]
        for i,vals in enumerate(xs):
            yy=i+np.linspace(-.12,.12,len(vals));a.scatter(vals,yy,s=28,facecolors='white',edgecolors=[OCHRE,TEAL][i],lw=1.65,zorder=4)
        a.axvline(15,ls='--',lw=1,c=MUTE);a.text(15.12,.42,'15 cm\ncriterion',va='center',size=7,color=MUTE)
        a.set(xlim=(10,19),ylim=(-.5,1.5),yticks=[0,1],yticklabels=['0.20','0.40'],xlabel='RMS position error (cm)',ylabel='Pitch-rate I gain')
        a=axis(f,[.605,.345,.315,.16],'c   Rotor thrust during bend')
        bt=b.since_release_s.values-22
        a.plot(bt,6*(b.m1.values+b.m3.values),c=NAVY,lw=1.5,label='Front pair')
        a.plot(bt,6*(b.m2.values+b.m4.values),c=OCHRE,lw=1.35,label='Rear pair')
        a.set(xlim=(0,14),ylim=(3.3,7.8),ylabel='Mean thrust (N)',xlabel='Time since bend onset (s)')
        a.legend(loc='upper left',fontsize=6.9,frameon=False)
        savecropped(f,'curves.png',[.034,.274,.961,.821])
        f.add_artist(plt.Line2D([.073,.926],[.264,.264],transform=f.transFigure,color=GRID,lw=.7))
        f.text(.073,.224,'Qualification flight',fontsize=11,weight='bold')
        ph=case['phases']
        for i,(key,value) in enumerate([('Hover RMS','%.2f cm'%(100*ph['hover']['rms_3d_m'])),('East translation','%.3f m'%case['displacement_east_m']),
                                   ('End error','%.2f cm'%(100*ph['translate']['endpoint_3d_m'])),('Peak bend','%.2f°'%case['max_total_curvature_deg'])]):
            xx=.075+.221*i;f.text(xx,.183,key,color=MUTE,fontsize=8);f.text(xx,.15,value,fontsize=14,weight='bold')
        f.text(.073,.093,'Bend-hold RMS was %.2f cm; the largest instantaneous deviation was %.2f cm.'%(100*ph['bend']['rms_3d_m'],100*ph['bend']['max_3d_m']),fontsize=8.8)
        f.text(.073,.065,'%s native motor samples match the HIL channels within %.1e in normalized output.'%(format(rec['overlap_native_motor_samples'],','),rec['max_native_normalized_motor_error']),fontsize=8.6)
        f.text(.073,.043,'This qualification covers an empty simulated workspace with nominal inertia, thrust and sensors.',fontsize=8.6)
        pdf.savefig(f);plt.close(f)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--figures-only',action='store_true');args=p.parse_args()
    out=args.root/'review';out.mkdir(exist_ok=True)
    if not args.figures_only:write_media(args.root,out)
    figures(args.root,out)
