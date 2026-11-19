"""Scientific plots and synchronized CAD/wrist replay from held-out native states."""
import subprocess,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, find_objects, label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from PIL import Image,ImageDraw,ImageFont
from round_health_flights import OUT
from round_health_evaluate import METHODS
from round_g1_model import make_plant
from round_g1_visuals import Camera
from round_f_rotor_replay import split_model,TURNS_PER_SIM_SECOND
from round_wrist_fusion import readj
from strain_health import angle_features
R=OUT/'review';R.mkdir(exist_ok=True);CASE='26112012-left'
INK='#22383d';GRAY='#53656d';GREEN='#177265';AMBER='#a96927';BLUE='#35698a';RED='#a14c37'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.65,'axes.edgecolor':'#aab7b9','text.color':INK,'axes.labelcolor':INK,'xtick.color':GRAY,'ytick.color':GRAY})
def save(fig,name):
    fig.savefig(R/(name+'.png'),dpi=180,facecolor='white');plt.close(fig)
    with Image.open(R/(name+'.png')) as im:im.convert('RGB').save(R/(name+'.webp'),quality=87,method=6)
def heading(a,txt):
    a.set_title(txt,loc='left',fontsize=9.9,fontweight='bold',pad=5)
def shade(ax):
    ax.axvspan(31.2,35.2,color='#e5eaeb',zorder=-2);ax.axhline(0,color='#9daaaa',lw=.65,zorder=-1)
    ax.grid(axis='y',color='#e1e7e6',lw=.5);ax.set_axisbelow(True);ax.set_xlim(23,41)
class Footage:
    def __init__(self,cam):
        folder=OUT/'runs'/CASE;self.cam=cam
        self.release=readj(folder/'run.json')['release_us'];ph=np.load(folder/'physics.npz');self.q=ph['qpos'];self.t=(ph['us']-self.release)/1e6
        data=pd.read_csv(folder/'hil.csv',usecols=['us','m1','m2','m3','m4'])
        self.w=(data.us.to_numpy()-self.release)/1e6;motor=data[['m1','m2','m3','m4']].to_numpy()
        self.spin=np.vstack((np.zeros(4),np.cumsum(np.sqrt(motor[:-1])*[1,1,-1,-1]*(np.diff(self.w)*2*np.pi*TURNS_PER_SIM_SECOND)[:,None],axis=0)))
        inp=OUT/'inputs/textured'/CASE;rows=[line.split() for line in (inp/'frames.txt').read_text().splitlines()]
        self.rgb=[inp/p for _,p,_ in rows];self.cap=np.array([float(t) for t,_,_ in rows])
        tab=pd.read_csv(OUT/'artifacts/slam/textured'/CASE/'occluded.csv')
        np.testing.assert_allclose(tab.t,self.cap);self.black=tab.occluded.to_numpy().astype(bool)
    def body(self,t):
        i=int(np.argmin(abs(self.t-t)));j=int(np.argmin(abs(self.w-t)));q=self.q[i]
        self.cam.cam.azimuth=109.;self.cam.cam.elevation=-14.
        return self.cam.image(q,self.spin[j],q[:3]+[.03,.0,-.17],.91)
    def wrist(self,t):
        i=max(0,np.searchsorted(self.cap+.2,t+1e-7,side='right')-1)
        return Image.new('RGB',(480,360),'black') if self.black[i] else Image.open(self.rgb[i]).convert('RGB')
def mechanism(foot):
    f=plt.figure(figsize=(9.1,4.42));gs=f.add_gridspec(2,2,left=.035,right=.979,bottom=.12,top=.895,
       width_ratios=[1.04,2.15],height_ratios=[1.,1.04],wspace=.32,hspace=.66)
    ax=f.add_subplot(gs[0,0]);im=np.asarray(foot.body(32.8))
    components,n=label(binary_dilation(im.min(axis=2)<145,iterations=3))
    ys,xs=find_objects(components)[np.argmax(np.bincount(components.ravel())[1:])]
    pad=14
    ax.imshow(im[max(0,ys.start-pad):min(len(im),ys.stop+pad),max(0,xs.start-pad):min(im.shape[1],xs.stop+pad)])
    ax.axis('off');heading(ax,'a   SUPER / Qingyu')
    sub=gs[1,0].subgridspec(1,2,wspace=.12)
    for j,tau in enumerate([26.,33.]):
        ax=f.add_subplot(sub[0,j]);ax.imshow(foot.wrist(tau));ax.set_xticks([]);ax.set_yticks([])
        ax.set_title('Clear' if j==0 else 'Covered',fontsize=8.7);ax.set_xlabel('%.1f s'%tau,fontsize=8.2)
        for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#a9b4b4')
    a=f.add_subplot(gs[0,1]);shade(a);heading(a,'b   Does the box show a zero shift?')
    for name,color,ls in [('healthy',GRAY,'--'),('memory',AMBER,'--'),('mixed',GREEN,'-')]:
        tr=np.load(OUT/'traces'/CASE/(name+'-occluded.npz'));t=tr['t'];p=tr['p']
        a.plot(t,p[:,1]+p[:,3],color=color,lw=1.15,ls=ls,label={'healthy':'Nominal','memory':'Memory','mixed':'Shift + memory'}[name])
    a.set_ylabel('Shift probability');a.set_ylim(-.055,1.06);a.set_yticks([0,.5,1]);a.legend(loc='center left',fontsize=7.0,frameon=False)
    a=f.add_subplot(gs[1,1]);shade(a);heading(a,'c   Is the bend still accurate?')
    tr=np.load(OUT/'traces'/CASE/'mixed-occluded.npz');truth=angle_features(np.load(OUT/'records'/CASE/'strain/reference.npz')['mocap_b'])[:,0]
    e=angle_features(tr['b'][METHODS.index('health')])[:,0]-truth;old=angle_features(tr['b'][METHODS.index('previous_teacher')])[:,0]-truth
    w=1.96*tr['reported_sigma'];a.fill_between(t,e-w,e+w,color='#b6d3cb',alpha=.72,label='Nominal 95% interval')
    a.plot(t,old,color=GRAY,ls='--',lw=1.0,label='Previous fusion');a.plot(t,e,color=GREEN,lw=1.35,label='Health-aware fusion')
    shown=(t>=23)&(t<=41);lo=min((e-w)[shown].min(),old[shown].min());hi=max((e+w)[shown].max(),old[shown].max())
    a.set_ylim(lo-.32*(hi-lo),hi+.08*(hi-lo))
    a.set_xlabel('Time after release (s)');a.set_ylabel('Bend error (deg)');a.legend(loc='lower left',fontsize=6.8,ncol=2,frameon=False)
    save(f,'mechanism')
def transfer():
    scores=readj(OUT/'flight-scores.json');labels=['Nominal','Zero shift','Material memory','Shift + memory','Directional shift']
    states=['healthy','offset','memory','mixed','differential'];fig,ax=plt.subplots(1,3,figsize=(9.25,3.55),sharey=True,
        gridspec_kw={'width_ratios':[1.08,1.08,.85]});fig.subplots_adjust(left=.17,right=.983,bottom=.19,top=.77,wspace=.18)
    ref=[('legacy','Original fixed noise',GRAY,'s'),('previous_teacher','Previous adaptive',BLUE,'o'),('health','Health-aware',GREEN,'D')]
    for k,(cond,ph,title) in enumerate([('natural','all','a   Clear camera'),('occluded','blind_window','b   Four-second loss')]):
        a=ax[k];heading(a,title)
        for i,(method,label,col,mk) in enumerate(ref):
            xx=[scores[g][cond][ph][method]['all']['theta_rmse_deg'] for g in states]
            a.scatter(xx,np.arange(5)+(i-1)*.17,marker=mk,s=27,color=col,label=label,zorder=3)
        a.set_xlim(-.35,12.2);a.set_xticks([0,4,8,12]);a.set_xlabel('Bend RMSE (deg)')
    a=ax[2];heading(a,'c   Interval coverage')
    xx=[100*scores[g]['occluded']['blind_window']['intervals']['reported_sigma']['coverage'] for g in states]
    for i,x in enumerate(xx):
        a.plot([0,x],[i,i],color='#aec8c3',lw=2);a.scatter([x],[i],s=26,color=GREEN,zorder=3)
    a.axvline(95,color=RED,ls=':',lw=1.1);a.set_xlim(-3,105);a.set_xticks([0,50,95]);a.set_xlabel('Blind interval covered (%)')
    for a in ax:
        a.grid(axis='x',color='#dfe6e5',lw=.55);a.set_axisbelow(True)
    ax[0].set_yticks(range(5));ax[0].set_yticklabels(labels);ax[0].set_ylim(4.55,-.55)
    fig.legend(*ax[0].get_legend_handles_labels(),ncol=3,loc='upper center',frameon=False,fontsize=8.7)
    save(fig,'transfer')
def video(foot):
    z=np.load(OUT/'traces'/CASE/'mixed-occluded.npz');truth=angle_features(np.load(OUT/'records'/CASE/'strain/reference.npz')['mocap_b'])[:,0]
    t=z['t'];vel=angle_features(z['b'][METHODS.index('health')])[:,0];p=z['p']
    font=lambda sz,b=False:ImageFont.truetype(findfont('DejaVu Sans:weight=bold' if b else 'DejaVu Sans'),sz)
    sm,md,bold=font(12),font(15),font(15,True)
    W,H=720,360;enc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','6','-i','-',
        '-an','-c:v','libx264','-preset','slow','-crf','31','-pix_fmt','yuv420p','-movflags','+faststart',str(R/'replay.mp4')],stdin=subprocess.PIPE)
    for ts in np.arange(23,40.99,1/6):
        i=np.searchsorted(t,ts);img=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(img)
        d.text((15,8),'Can four strain gauges distinguish drift from material memory?',font=bold,fill=INK)
        d.text((15,35),'Aircraft and arm',font=sm,fill=GRAY);d.text((277,35),'Wrist RGB-D',font=sm,fill=GRAY)
        d.text((490,35),'Shape estimate',font=sm,fill=GRAY)
        body=foot.body(ts).crop((40,48,442,450)).resize((253,253));img.paste(body,(12,56))
        img.paste(foot.wrist(ts).resize((191,144)),(277,56))
        d.rectangle((276,55,468,201),outline='#bac7c7')
        d.text((277,216),'Time after release',font=sm,fill=GRAY);d.text((277,236),'%.2f s'%ts,font=bold,fill=INK)
        d.text((490,65),'Estimate   %5.1f deg'%vel[i],font=md,fill=GREEN);d.text((490,90),'Mocap      %5.1f deg'%truth[i],font=md,fill=INK)
        d.text((490,132),'Probability of zero shift',font=sm,fill=GRAY)
        v=float(p[i,1]+p[i,3]);d.rectangle((490,156,690,164),fill='#e2e9e8');d.rectangle((490,156,int(490+200*v),164),fill=BLUE)
        d.text((490,176),'%.2f'%v,font=bold,fill=BLUE)
        d.text((490,220),'Nominal 95% half-width',font=sm,fill=GRAY);d.text((490,241),'%.1f deg'%(1.96*z['reported_sigma'][i]),font=md,fill=GREEN)
        d.text((16,326),'PX4 SITL / MuJoCo   |   left-offset gate pass   |   injected zero shift + material memory',font=sm,fill=GRAY)
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    if enc.wait():raise RuntimeError('Cannot encode synchronized replay')
if __name__=='__main__':
    transfer()
    with tempfile.TemporaryDirectory() as tmp:
        m,axes,_=split_model(make_plant(),Path(tmp));cam=Camera(m,axes)
        try:
            footage=Footage(cam);mechanism(footage);video(footage)
        finally:cam.ren.close()
