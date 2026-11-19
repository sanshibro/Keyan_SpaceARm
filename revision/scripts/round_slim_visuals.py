"""Journal-size material, readout and estimator figures from the registered replay."""
import subprocess,tempfile
from pathlib import Path
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from slim_sensor import SlimGaugeRig
from round_slim_box import OUT,SPEC
from round_g1_model import SimArmBridge,make_plant
from round_g1_online_visuals import Flight
from round_g1_visuals import Camera
from round_f_rotor_replay import split_model,font
from round_wrist_visuals import side_frame
from round_wrist_fusion import readj,vector_angles
from round_shape_pixels import OUTPUT as OLD

R=OUT/'review';CASE='241119-online_pause'
C=['#226b60','#b47837','#526f9b','#92536d'];INK='#243940';MUTED='#5b686c'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#a8b0ae','axes.linewidth':.7,'text.color':INK,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED})
def title(a,s):a.set_title(s,loc='left',fontweight='bold',fontsize=10.6,pad=8)
def finish(fig,name):
    fig.savefig(R/(name+'.png'),dpi=170,facecolor='white')
    fig.savefig(R/(name+'.pdf'),facecolor='white');plt.close(fig)
    with Image.open(R/(name+'.png')) as im:im.convert('RGB').save(R/(name+'.webp'),quality=85,method=6)
def plate(cam):
    fig=plt.figure(figsize=(10.8,3.5));g=fig.add_gridspec(1,3,left=.023,right=.984,top=.86,bottom=.20,width_ratios=[1.02,1,1.04],wspace=.32)
    a,b,c=[fig.add_subplot(g[i]) for i in range(3)]
    m,d=cam.m,cam.d;rig=SlimGaugeRig(m,SPEC);d.qpos[:]=m.qpos0
    SimArmBridge().set_kinematic_shape(m,d,np.deg2rad(91),np.deg2rad(5));mujoco.mj_kinematics(m,d)
    q=d.qpos.copy();center=(d.body('seg0').xpos+d.body('tool').xpos)/2
    # Only the fitting close-up is isolated from the static gate and floor.
    groups=m.geom_group.copy()
    for i in range(m.ngeom):
        if m.geom_bodyid[i]==0:m.geom_group[i]=5
    cam.cam.azimuth=118.;cam.cam.elevation=-13.
    im=cam.image(q,np.zeros(4),center,.40);a.imshow(im);a.axis('off')
    a.set_xlim(0,480);a.set_ylim(480,0)
    points=rig.paths(cam.d)
    for j in range(4):
        pts=np.array([cam.project(x) for x in points[:,j,:]])
        a.plot(pts[:,0],pts[:,1],color=C[j],lw=2.7,solid_capstyle='round')
    target=cam.project(points[:,np.argmax([cam.project(points[1,k])[0] for k in range(4)]),:].mean(0))
    a.annotate('Midspan',xy=target,xytext=(.91,.15),textcoords='axes fraction',ha='right',fontsize=8.5,
        color=INK,bbox=dict(facecolor='white',edgecolor='none',pad=1.8),arrowprops=dict(arrowstyle='-',color=INK,lw=.8))
    m.geom_group[:]=groups;cam.cam.azimuth=98.;cam.cam.elevation=-7.
    title(a,'a   Surface paths on the arm')
    op=np.array(SPEC['resistance_curve_25C']['engineering_strain'])*100
    rr=100*np.array(SPEC['resistance_curve_25C']['relative_resistance_change'])
    b.plot(op,rr,'o-',color=C[0],ms=3.7,lw=1.4,zorder=4)
    lim=readj(OUT/'sweep.json')['p30']
    b.axvspan(lim['min_120_percent'],lim['max_120_percent'],color='#deeee7',zorder=0)
    b.axvline(30,color='#53696b',lw=.85,ls='--')
    title(b,'b   Material response at 25 °C');b.set(xlim=(-2,102),ylim=(-3,170),xticks=[0,25,50,75,100],yticks=[0,50,100,150],xlabel='Tensile strain (%)',ylabel='Resistance change (%)')
    b.text(.08,.88,'32.5 mm mount',transform=b.transAxes,color=C[0],fontsize=8);b.text(.08,.80,'120° sweep range',transform=b.transAxes,color=C[0],fontsize=8)
    sw=np.load(OUT/'sweep.npz');angles=sw['bend_deg'];ext=sw['extension']
    for j,style in [(0,':'),(2,'-'),(3,'--')]:
        ee=100*ext[:,:,j];col={0:'#777a79',2:C[0],3:C[1]}[j]
        c.plot(angles,ee.min((1,2)),style,color=col,lw=1.65,label=['25 mm','','32.5 mm','35 mm'][j])
        c.plot(angles,ee.max((1,2)),style,color=col,lw=1.65)
    c.axhline(0,color='#7e8889',lw=.75);c.axhspan(-24,0,color='#f6eee8',zorder=0)
    title(c,'c   Inner and outer strain');c.set(xlim=(0,120),xticks=[0,30,60,90,120],ylim=(-23,72),yticks=[-20,0,20,40,60],xlabel='Total arm bend (deg)',ylabel='Surface strain (%)')
    c.legend(loc='upper left',fontsize=7.6,frameon=False,handlelength=2.2,handletextpad=.45,labelspacing=.2)
    for ax in (b,c):ax.grid(axis='y',color='#dfe6e3',lw=.5)
    fig.text(.042,.04,'Four paths at 90° intervals; coloured lines show material locations.',fontsize=8,color=MUTED)
    fig.text(.615,.04,'Envelopes span four channels and 24 bend planes.',fontsize=8,color=MUTED)
    finish(fig,'mechanism')
def curves():
    src=OUT/'primary'/CASE
    z=np.load(src/'occluded.npz');sig=np.load(src/'strain.npz');t=z['t'];v=vector_angles(z['raw_b'])[0]
    ref=vector_angles(z['truth'])[0];fault=np.load(OUT/'return_offset'/CASE/'occluded.npz');drift=vector_angles(fault['raw_b'][3])[0]-ref
    fig,ax=plt.subplots(3,1,figsize=(9.9,5.5),sharex=True,gridspec_kw={'height_ratios':[1.15,1,.95]})
    fig.subplots_adjust(left=.103,right=.983,bottom=.095,top=.91,hspace=.66)
    for a in ax:
        a.axvspan(31.2,35.2,color='#efe4ce',lw=0);a.grid(axis='y',color='#d9e2df',lw=.55);a.set_xlim(14,46)
    for j in range(4):ax[0].plot(sig['t'],sig['adc'][:,j],color=C[j],lw=1.05,label='CH%d'%(j+1))
    ax[0].set_ylabel('Box output (counts)');ax[0].set_ylim(bottom=0);title(ax[0],'a   Four sensor channels at the measured acquisition cadence')
    ax[0].legend(ncol=4,loc='lower right',bbox_to_anchor=(1,1.025),frameon=False,columnspacing=.9,fontsize=8)
    ax[1].plot(t,ref,color=INK,label='Virtual mocap',lw=3,alpha=.8)
    ax[1].plot(t,v[2],color=C[2],ls='--',label='ORB-SLAM3',lw=1.45)
    ax[1].plot(t,v[3],color=C[0],label='Fusion',lw=1.1)
    title(ax[1],'b   Bend magnitude against independent truth');ax[1].set_ylabel('Angle (deg)')
    ax[1].legend(ncol=3,loc='lower right',bbox_to_anchor=(1,1.025),frameon=False,fontsize=8)
    ax[2].plot(t,drift,color=C[1],lw=1.3,label='Zero-return offset applied')
    ax[2].plot(t,v[3]-ref,color=C[0],lw=1.1,label='Nominal zero')
    ax[2].axhline(0,color='#828e8a',lw=.6,zorder=0)
    ax[2].set_ylabel('Error (deg)');ax[2].set_xlabel('Time after release (s)')
    title(ax[2],'c   Sensitivity to a changed zero');ax[2].legend(ncol=2,loc='lower right',bbox_to_anchor=(1,1.025),frameon=False,fontsize=8)
    ax[0].text(33.2,.86,'Visual loss',transform=ax[0].get_xaxis_transform(),ha='center',fontsize=7.7,color='#69532c')
    finish(fig,'curves')
def movie(cam):
    z=np.load(OUT/'primary'/CASE/'occluded.npz');sig=np.load(OUT/'primary'/CASE/'strain.npz')
    stream=np.load(OLD/'slam/textured'/CASE/'wrists.npz');flight=Flight(CASE)
    ref=vector_angles(z['truth'])[0];est=vector_angles(z['raw_b'][3])[0]
    head=font(21,True);body=font(15);small=font(13)
    W,H=704,402;fps=4
    cmd=['ffmpeg','-nostdin','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','29','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(R/'comparison.mp4')]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for now in np.arange(14,46,1/fps):
            canvas=Image.new('RGB',(W,H),'white');draw=ImageDraw.Draw(canvas)
            draw.text((16,8),'臂体中段四路应变与腕部视觉',font=head,fill=INK)
            draw.text((591,13),'t = %04.1f s'%now,font=body,fill=MUTED)
            frame,_,_=side_frame(cam,flight,now,1.05);frame=frame.crop((0,65,480,406)).resize((278,198),Image.Resampling.LANCZOS if hasattr(Image,'Resampling') else Image.LANCZOS);canvas.paste(frame,(14,62))
            k=int(np.argmin(abs(stream['t']-now)));src=Image.fromarray(stream['rgb'][k])
            if 31<=now<35:src=Image.new('RGB',src.size,'black')
            src=src.resize((254,190));canvas.paste(src,(316,65))
            draw.text((16,38),'已记录整机姿态',font=small,fill=MUTED);draw.text((316,38),'腕部 RGB-D 彩色帧',font=small,fill=MUTED)
            idx=int(np.searchsorted(sig['t'],now,side='right')-1);tick=int(np.searchsorted(z['t'],now,side='right')-1)
            draw.text((587,67),'采集盒读数',font=small,fill=MUTED)
            for j in range(4):draw.text((587,99+31*j),'CH%d  %4d'%(j+1,sig['adc'][idx,j]),font=small,fill=C[j])
            draw.text((18,272),'融合弯角  %.1f°     虚拟动捕  %.1f°'%(est[tick],ref[tick]),font=body,fill=INK)
            state='原地图视觉可用' if np.isfinite(z['raw_b'][2,tick]).all() else '原地图视觉无输出'
            draw.text((333,272),state,font=body,fill=C[2] if '可用' in state else C[1])
            xx=lambda s:18+(s-14)*664/32
            yy=lambda y:367-y*.64
            draw.rectangle((xx(31.2),309,xx(35.2),370),fill='#ede4cf')
            draw.line((18,370,682,370),fill='#bcc6c3')
            sel=np.flatnonzero((z['t']>=14)&(z['t']<=now))[::4]
            if len(sel)>1:
                xy=lambda y:[(xx(z['t'][i]),yy(y[i])) for i in sel]
                draw.line(xy(ref),fill='#303e45',width=3);draw.line(xy(est),fill=C[0],width=1)
            draw.text((16,377),'14 s',font=small,fill=MUTED);draw.text((392,377),'相机遮挡',font=small,fill=MUTED);draw.text((650,377),'46 s',font=small,fill=MUTED)
            proc.stdin.write(np.asarray(canvas).tobytes())
    finally:proc.stdin.close()
    if proc.wait():raise RuntimeError('ffmpeg could not finish')
    subprocess.run(['ffmpeg','-nostdin','-loglevel','error','-y','-i',str(R/'comparison.mp4'),
        '-vf','fps=3,scale=560:-2:flags=lanczos','-c:v','libx264','-crf','33','-preset','slow',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(R/'replay.mp4')],check=True)
if __name__=='__main__':
    R.mkdir(exist_ok=True);curves()
    with tempfile.TemporaryDirectory() as td:
        m,axes,_=split_model(make_plant(),Path(td));cam=Camera(m,axes)
        try:plate(cam);movie(cam)
        finally:cam.ren.close()
