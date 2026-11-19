"""Publication-scale rendering of one registered flight and its causal estimator outputs."""
from pathlib import Path
import subprocess,tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from round_f_rotor_replay import split_model,font
from round_g1_model import make_plant
from round_g1_visuals import Camera
from round_g1_online_visuals import Flight
from round_shape_pixels import OUTPUT
from round_wrist_fusion import readj,vector_angles

CASE='241119-online_pause';R=OUTPUT/'review';SRC=OUTPUT/'slam/textured'/CASE
INK='#263d43';GREY='#576970';GREEN='#237465';BLUE='#466a91';RED='#a9553d';GOLD='#8d7344'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                    'axes.edgecolor':'#a8b2b1','axes.linewidth':.7,'text.color':INK,'axes.labelcolor':INK,
                    'xtick.color':GREY,'ytick.color':GREY,'figure.facecolor':'white','savefig.facecolor':'white'})

def state(row,anchor):
    if row.occluded:return 'occluded'
    if row.state!=2:return 'untracked'
    if row.map_id!=anchor:return 'new'
    return 'mapped'
LABEL={'mapped':'Initial map','occluded':'Optical blackout','new':'Unregistered new map','untracked':'Tracking lost'}
ZH={'mapped':'初始地图内跟踪','occluded':'相机光学遮挡','new':'已新建地图，坐标尚未对齐','untracked':'视觉跟踪中断'}

def wrist(stream,table,t,anchor):
    i=int(np.argmin(abs(stream['t']-t)));stamp=stream['t'][i]
    row=table.iloc[np.argmin(abs(table.t.to_numpy()-stamp))]
    label=state(row,anchor)
    picture=Image.fromarray(stream['rgb'][i])
    if row.occluded:picture=Image.new('RGB',picture.size,'black')
    return picture,label,row

def side_frame(cam,flight,t,span):
    index=int(np.argmin(abs(flight.t-t)))
    center=(flight.a['qpos'][index,0]+1.)/2
    return flight.frame(cam,t,[center,0.,1.98],span)

def plate(cam,flight,stream,tab,anchor):
    fig,axes=plt.subplots(2,3,figsize=(10.25,4.95))
    fig.subplots_adjust(left=.026,right=.985,top=.915,bottom=.115,hspace=.02,wspace=.055)
    for col,t in enumerate((28.,33.,38.)):
        im,j,k=side_frame(cam,flight,t,1.30)
        axes[0,col].imshow(np.asarray(im)[74:398]);axes[0,col].axis('off')
        axes[0,col].set_title(chr(97+col)+'   '+['Before loss','Camera covered','Camera uncovered'][col]+'   /   %.0f s'%t,
                              loc='left',fontsize=10,fontweight='bold',pad=5)
        image,label,row=wrist(stream,tab,t,anchor)
        axes[1,col].imshow(image);axes[1,col].axis('off')
        axes[1,col].text(.0,-.08,LABEL[label],transform=axes[1,col].transAxes,color=GREEN if label=='mapped' else RED,fontsize=9.8)
    fig.text(.036,.022,'Upper: recorded vehicle and arm, common side projection.     Lower: wrist RGB-D image supplied to ORB-SLAM3.     Flight 241119 / online timing.',fontsize=8,color=GREY)
    fig.savefig(R/'mechanism.png',dpi=150);plt.close(fig)

def curves(z,tab,anchor):
    t=z['t'];b=vector_angles(z['b'])[0];truth=vector_angles(z['truth'])[0];idx=(t>=14)&(t<46)
    clock=t[idx];ref=truth[idx]
    fig,axes=plt.subplots(3,1,figsize=(9.6,5.55),sharex=True,gridspec_kw={'height_ratios':[1.15,1,.64]})
    fig.subplots_adjust(left=.10,right=.973,top=.89,bottom=.12,hspace=.60)
    def span(ax):
        ax.axvspan(31.2,35.2,color='#e9e0ce',alpha=.64,linewidth=0)
        ax.axvline(35.2,color='#b9a98a',ls=':',lw=.85)
        ax.set_xlim(14,46);ax.grid(axis='y',lw=.48,color='#dce2e1')
    span(axes[0]);span(axes[1])
    axes[0].plot(clock,ref,color=INK,lw=2.4,label='Virtual mocap')
    axes[0].plot(clock,b[0,idx],color=GOLD,lw=.8,ls=':',label='Strain')
    axes[0].plot(clock,b[2,idx],color=BLUE,lw=1.05,ls='--',label='ORB-SLAM3')
    axes[0].plot(clock,b[3,idx],color=GREEN,lw=1.1,label='Fusion')
    axes[0].set_ylabel('Bend (deg)');axes[0].set_yticks([0,35,70])
    axes[0].set_title('a   Bend against independent virtual motion capture',loc='left',fontweight='bold',fontsize=10)
    axes[0].legend(ncol=4,frameon=False,loc='lower right',bbox_to_anchor=(1,1.12),fontsize=8,columnspacing=1.1)
    for i,col,lab,ls in [(2,BLUE,'ORB-SLAM3','--'),(0,GOLD,'Strain',':'),(3,GREEN,'Fusion','-')]:
        axes[1].plot(clock,b[i,idx]-ref,color=col,ls=ls,lw=1.1 if i!=0 else .85)
    axes[1].axhline(0,color='#aebdb9',lw=.65);axes[1].set_ylabel('Error (deg)')
    axes[1].set_title('b   Signed error at delivery time',loc='left',fontweight='bold',fontsize=10)
    axes[2].set_title('c   Which observations can update the estimate?',loc='left',fontweight='bold',fontsize=10)
    axes[2].set_yticks([0,1]);axes[2].set_yticklabels(['Strain','SLAM'])
    axes[2].set_ylim(-.63,1.55)
    axes[2].broken_barh([(14,32)],(-.13,.26),facecolors=GOLD)
    colors={'mapped':BLUE,'occluded':'#dfd3b9','new':'#e7ebeb','untracked':'#dae0e0'}
    spans=[(row.t+.2,state(row,anchor)) for row in tab.itertuples()]
    start,label=spans[0]
    for end,next_label in spans[1:]+[(46.,None)]:
        if next_label!=label:
            left=max(14.,start);right=min(end,46.)
            if right>left:axes[2].broken_barh([(left,right-left)],(.85,.3),facecolors=colors[label])
            start,label=end,next_label
    axes[2].text(33.2,1.30,'Blackout',ha='center',fontsize=7.8,color=GREY)
    axes[2].text(40.6,1.30,'New map',ha='center',fontsize=7.8,color=GREY)
    axes[2].set_xlabel('Time after release (s)');axes[2].set_xticks([15,20,25,30,35,40,45])
    fig.savefig(R/'curves.png',dpi=157);plt.close(fig)

def film(cam,flight,stream,tab,anchor,z):
    W,H=900,520;fps=5
    encoder=['ffmpeg','-nostdin','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(fps),
             '-i','-','-an','-c:v','libx264','-preset','fast','-crf','27','-maxrate','350k','-bufsize','550k',
             '-pix_fmt','yuv420p','-movflags','+faststart',str(R/'comparison.mp4')]
    proc=subprocess.Popen(encoder,stdin=subprocess.PIPE);head=font(23,True);body=font(17);sub=font(15)
    gt=vector_angles(z['truth'])[0];be=vector_angles(z['b'])[0];axis=z['t']
    try:
        for t in np.arange(14.,46.,1./fps):
            canvas=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(canvas)
            d.text((22,11),'视觉中断时，柔性臂弯曲量由谁观测？',font=head,fill=INK)
            d.text((754,14),'%.1f s'%t,font=head,fill=INK);d.line((22,48,878,48),fill='#c9d1d0')
            d.text((22,57),'SUPER 机架与柔性臂 · 侧视剖面',font=body,fill=INK)
            d.text((465,57),'末端 RGB-D · 实际送入跟踪器的图像',font=body,fill=INK)
            scene,_,_=side_frame(cam,flight,t,1.48)
            canvas.paste(scene.resize((416,416),Image.LANCZOS).crop((0,77,416,350)),(23,85))
            im,label,row=wrist(stream,tab,t,anchor);canvas.paste(im.resize((388,273),Image.LANCZOS),(468,85))
            d.text((465,365),ZH[label],font=body,fill=GREEN if label=='mapped' else RED)
            k=int(np.argmin(abs(axis-t)));d.text((24,365),'动捕真值 %.1f°    融合 %.1f°'%(gt[k],be[3,k]),font=body,fill=INK)
            d.text((24,397),'弯曲量',font=sub,fill=GREY)
            def line(y,col):
                xy=[(115+(axis[j]-14)*23,482-y[j]*.94) for j in np.where((axis>=14)&(axis<=t)&np.isfinite(y))[0][::3]]
                if len(xy)>1:d.line(xy,fill=col,width=2)
            d.rectangle((115+(31.2-14)*23,404,115+(35.2-14)*23,482),fill='#f0e9d9')
            line(gt,INK);line(be[3],GREEN)
            d.line((115,483,852,483),fill='#bbc5c3')
            d.text((23,490),'仿真飞行记录重放  ·  遮挡 31–35 s  ·  曲线依输出时间绘制，视觉通道设定延迟 0.20 s',font=sub,fill=GREY)
            proc.stdin.write(canvas.tobytes())
    finally:proc.stdin.close()
    if proc.wait():raise RuntimeError('video encoder failed')

def main():
    R.mkdir(exist_ok=True)
    z=dict(np.load(str(SRC/'occluded-estimate.npz')));stream=np.load(str(SRC/'wrists.npz'))
    tab=pd.read_csv(SRC/'occluded.csv');anchor=readj(SRC/'occluded-diagnostics.json')['initial_map']
    curves(z,tab,anchor)
    with tempfile.TemporaryDirectory() as tmp:
        m,axes,_=split_model(make_plant(),Path(tmp));cam=Camera(m,axes);flight=Flight(CASE)
        try:plate(cam,flight,stream,tab,anchor);film(cam,flight,stream,tab,anchor,z)
        finally:cam.ren.close()
if __name__=='__main__':main()
