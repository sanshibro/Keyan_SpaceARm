"""Observed trigger poses, synchronized playback and paired outcome figures."""
import argparse
import json
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from round_f_plant import CHANNELS
from round_f_rotor_replay import split_model,font,TURNS_PER_SIM_SECOND
from round_g1_visuals import Camera
from round_g1_model import make_plant,TARGET
from round_g1_online_run import DEST,P
from round_g1_online_audit import applied,records

OUT=DEST/'review';COL={'clock':'#9c6550','phase':'#44718d','online':'#247668'}
NAME={'clock':'Fixed clock','phase':'Route phase','online':'Online timing'}
CN={'clock':'固定时钟','phase':'指令进度','online':'状态反馈'}
KEYS=('clock','phase','online');INK='#263d43';GRAY='#54676e'
plt.rcParams.update({'font.family':'DejaVu Serif','axes.spines.top':False,'axes.spines.right':False,
                     'axes.edgecolor':'#9daaa9','axes.linewidth':.7,'text.color':INK,'axes.labelcolor':INK,
                     'xtick.color':GRAY,'ytick.color':GRAY,'font.size':9})


class Flight:
    def __init__(self,case):
        self.case=case;self.seed,mode=case.split('-');self.method,self.condition=mode.split('_')
        folder=DEST/'runs'/case;self.a=dict(np.load(str(folder/'physics.npz')))
        w=pd.read_csv(folder/'hil.csv');self.release=json.loads((folder/'run.json').read_text())['release_us']
        self.t=(self.a['us']-self.release)/1e6;self.wt=(w.us.to_numpy()-self.release)/1e6
        motor=w[['m1','m2','m3','m4']].values
        self.phase=np.vstack([np.zeros(4),np.cumsum(np.sqrt(motor[:-1])*[1,1,-1,-1]*
            (2*np.pi*TURNS_PER_SIM_SECOND*np.diff(self.wt))[:,None],axis=0)])
        self.measured=w.net_deg.to_numpy();self.u=w.request_deg.to_numpy()
        self.rows=records(folder/'timing.jsonl')
        dec={r['action']:r['t'] for r in self.rows if r['action']!='retain'}
        self.actions=applied(self.method,self.condition,dec)
        self.tr=pd.read_csv(DEST/'traces'/(case+'.csv')) if (DEST/'traces'/(case+'.csv')).exists() else None

    def frame(self,cam,t,center,span):
        j=int(np.argmin(abs(self.t-t)));k=int(np.argmin(abs(self.wt-t)))
        return cam.image(self.a['qpos'][j],self.phase[k],center,span),j,k


def preview(cam,f):
    fig=plt.figure(figsize=(9.1,4.7))
    for i,(key,heading) in enumerate([('bend','Begin bending'),('return','Begin retraction')]):
        t=f.actions[key];j=int(np.argmin(abs(f.t-t)));pos=f.a['qpos'][j,:3]
        im,_,k=f.frame(cam,t,[pos[0],0.,pos[2]-.16],.92)
        ax=fig.add_axes([.035+i*.303,.26,.265,.62]);ax.imshow(im);ax.axis('off')
        fig.text(.037+i*.303,.93,chr(97+i)+'   '+heading,fontsize=12,fontweight='bold')
        fig.text(.038+i*.303,.21,'t = %.1f s     x = %.2f m'%(t,pos[0]),fontsize=10.5,color=COL['online'])
        fig.text(.038+i*.303,.15,'Measured bend   %.1f°'%f.measured[k],fontsize=9,color=GRAY)
    ax=fig.add_axes([.692,.23,.275,.58])
    mask=(f.wt>=11)&(f.wt<=50)
    ax.plot(f.wt[mask],f.u[mask],color='#76838b',ls='--',lw=1.35,label='Command')
    ax.plot(f.wt[mask],f.measured[mask],color=COL['online'],lw=1.7,label='Measured')
    for k in ('bend','return'):ax.axvline(f.actions[k],color='#9ab4ab',ls=':',lw=.9)
    ax.set_xlim(11,50);ax.set_ylim(-3,84);ax.set_xticks([15,25,35,45]);ax.set_yticks([0,25,50,75])
    ax.set_xlabel('Time after release (s)');ax.set_ylabel('Bend (°)')
    ax.grid(axis='y',color='#e3e8e6',lw=.55);ax.legend(frameon=False,fontsize=8,loc='lower center',bbox_to_anchor=(.53,1))
    fig.text(.685,.93,'c   Applied arm command',fontsize=12,fontweight='bold')
    fig.text(.04,.044,'Actual MuJoCo states at the two accepted transitions. Local views share orientation and scale.     Seed '+f.seed,fontsize=8,color=GRAY)
    fig.savefig(OUT/('trigger-'+f.seed+'.png'),dpi=195);plt.close(fig)


def comparison_plate(cam,fl):
    fig=plt.figure(figsize=(11.1,4.7))
    for j,key in enumerate(KEYS):
        f=fl[key];t=f.actions['return']
        im,i,k=f.frame(cam,t,[1.17,0.,1.99],1.38)
        ax=fig.add_axes([.032+.324*j,.226,.298,.61])
        ax.imshow(im);ax.axis('off')
        pt=cam.project(f.a['qpos'][i,:3]);ax.scatter(*pt,s=10,c=COL[key],edgecolors='white',linewidth=.4)
        rear=100*np.interp(t,f.tr.t,f.tr.rear_m)
        fig.text(.041+.324*j,.919,chr(97+j)+'   '+NAME[key],fontsize=12,fontweight='bold')
        fig.text(.043+.324*j,.145,'Rear relative to far face   %+.1f cm'%rear,fontsize=10.3,color=COL[key])
        fig.text(.043+.324*j,.088,'Retraction begins at %.1f s'%t,fontsize=9,color=GRAY)
    fig.text(.041,.023,'Same camera and aperture in all three panels. Negative rear distance means that part of the assembly is still inside the gate plane.',fontsize=8,color=GRAY)
    fig.savefig(OUT/'release.png',dpi=200);plt.close(fig)


def curves(fl,metrics):
    fig,axes=plt.subplots(2,2,figsize=(10.8,6.05));fig.subplots_adjust(left=.085,right=.972,top=.9,bottom=.19,hspace=.58,wspace=.29)
    for j,(col,title,yl) in enumerate([('rear_m','When does the full assembly clear the gate?','Rear to far face (cm)'),
            ('arm_tool_m','Arm and tool clearance','Lower bound (cm)'),('bend_deg','Measured arm response','Bend (°)')]):
        ax=axes.flat[j];ax.set_title(chr(97+j)+'   '+title,fontweight='bold',loc='left',fontsize=9.8)
        for method in KEYS:
            f=fl[method];tr=f.tr
            q=(tr.t>=24)&(tr.t<=46)
            scale=1 if col=='bend_deg' else 100
            ax.plot(tr.t[q].to_numpy(),tr[col][q].to_numpy()*scale,color=COL[method],lw=1.55,
                    ls='--' if method=='clock' else '-.' if method=='phase' else '-')
            when=f.actions['return'];value=float(np.interp(when,tr.t,tr[col]))*scale
            ax.scatter([when],[value],color=COL[method],s=20,edgecolors='white',linewidth=.5,zorder=5)
        if col=='rear_m':ax.axhline(0,color='#889796',lw=.75);ax.set_ylim(-115,86)
        elif col=='arm_tool_m':ax.axhline(7,color='#9b6852',lw=.8,ls=':');ax.set_ylim(0,52)
        else:ax.set_ylim(0,84)
        ax.grid(axis='y',color='#e4e9e7',lw=.55);ax.set_ylabel(yl);ax.set_xlabel('Time after release (s)');ax.set_xlim(24,46)
    ax=axes.flat[3];ax.set_title('d   Matched minima under the route pause',fontweight='bold',loc='left',fontsize=9.8)
    for seed in sorted(metrics.seed.unique()):
        sel=metrics[(metrics.seed==seed)&(metrics.condition=='pause')].set_index('method')
        yy=[sel.loc[k,'arm_convex_cm'] for k in KEYS]
        ax.plot([0,1,2],yy,lw=.85,color='#99a5a6')
        ax.scatter([0,1,2],yy,color=[COL[k] for k in KEYS],s=20,edgecolors='white',linewidth=.5,zorder=4)
    ax.set_xticks([0,1,2]);ax.set_xticklabels(['Clock','Phase','Online'])
    ax.set_ylabel('Convex clearance (cm)');ax.set_xlim(-.18,2.18);ax.grid(axis='y',color='#e4e9e7',lw=.55)
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=COL[k],ls='--' if k=='clock' else '-.' if k=='phase' else '-',lw=1.8) for k in KEYS]
    fig.legend(handles+ [Line2D([],[],color='#657772',ls='',marker='o',markersize=4)],
               [NAME[k] for k in KEYS]+['Start of retraction'],loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.50,.013),fontsize=9)
    fig.savefig(OUT/'curves.png',dpi=196);plt.close(fig)


def video(cam,fl):
    W,H=1456,644;fps=15
    cmd=['ffmpeg','-nostdin','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(fps),
         '-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'comparison.mp4')]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);ff=[font(26,True),font(20,True),font(16)]
    ts=np.arange(19.,50.001,1/fps)
    try:
        for t in ts:
            board=Image.new('RGB',(W,H),'white');pen=ImageDraw.Draw(board)
            pen.text((24,11),'门前停顿后，何时可以收臂？',font=ff[0],fill=INK)
            pen.text((1200,18),'离地后  %05.2f s'%t,font=ff[1],fill=INK);pen.line((24,59,1432,59),fill='#bdc9c8')
            for j,key in enumerate(KEYS):
                f=fl[key];x=24+477*j;im,i,k=f.frame(cam,t,[.99,0.,2.02],2.28)
                w=450
                im=im.resize((w,w),Image.LANCZOS).crop((0,56,w,373));d=ImageDraw.Draw(im)
                pix=lambda p:tuple(cam.project(p)*w/480-[0,56])
                trail=f.a['tcp'][max(0,i-120):i+1:3]
                if len(trail)>1:d.line([pix(q) for q in trail],fill=COL[key],width=2)
                px,py=pix(TARGET);d.line((px-6,py,px+6,py),fill='#a14f3c',width=2);d.line((px,py-6,px,py+6),fill='#a14f3c',width=2)
                board.paste(im,(x,113));pen.text((x,74),CN[key],font=ff[1],fill=COL[key])
                idx=int(np.argmin(abs(f.tr.t.to_numpy()-t)));row=f.tr.iloc[idx]
                pen.text((x,454),'机身后缘距门后表面   %+.1f cm'%(100*row.rear_m),font=ff[2],fill=INK)
                pen.text((x,485),'臂与夹爪净空下界   %.1f cm'%(100*row.arm_tool_m),font=ff[2],fill=INK)
                pen.text((x,516),'实际弯角   %.1f°'%row.bend_deg,font=ff[2],fill=INK)
                state='已发出收臂指令' if t>=f.actions['return'] else '保持或进入过门臂形'
                pen.text((x,548),state,font=ff[1],fill=COL[key])
            pen.line((25,595,1431,595),fill='#c5cecb')
            pen.text((24,603),'共同世界机位、相同机身指令和 52 s 截止时间。后缘距离小于零表示整机尚未完全越门。桨叶相位按推力降速显示。',font=ff[2],fill=GRAY)
            if abs(t-36.)<1e-6:board.save(OUT/'matched.png')
            proc.stdin.write(board.tobytes())
    finally:proc.stdin.close()
    if proc.wait():raise RuntimeError('Video encoding failed')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview');ap.add_argument('--video',action='store_true');a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        m,axes,_=split_model(make_plant(),Path(temp));cam=Camera(m,axes)
        try:
            if a.preview:preview(cam,Flight(a.preview));return
            fl={k:Flight(str(P['visual_seed'])+'-'+k+'_pause') for k in KEYS}
            preview(cam,fl['online']);comparison_plate(cam,fl);curves(fl,pd.read_csv(DEST/'metrics.csv'))
            if a.video:video(cam,fl)
        finally:cam.ren.close()


if __name__=='__main__':main()
