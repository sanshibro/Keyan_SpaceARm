"""Render recorded, integrated fine-time-step states at a fixed scale (CAD, not interpolation)."""
import subprocess
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from round_e1_run import DEFAULT
from round_e1_model import build, neutral

def render():
    raw={n:np.load(str(DEFAULT/('n%02d_dt0500us.npz'%n))) for n in (8,16,24)}
    src=raw[8]
    m,_=build(8,.0005)
    d=neutral(m)
    h=w=620
    renderer=mujoco.Renderer(m,height=h,width=w)
    cam=mujoco.MjvCamera();cam.orthographic=1
    cam.lookat[:]=[.045,0,1.28];cam.distance=1.1;cam.azimuth=108;cam.elevation=-9
    m.vis.global_.fovy=.86
    opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[3]=0;opt.sitegroup[:]=0
    def frame(i):
        d.qpos[:]=src['qpos'][i];mujoco.mj_forward(m,d)
        renderer.update_scene(d,camera=cam,scene_option=opt)
        return renderer.render().copy()
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(8,3.12))
    ts=[0,4,8]
    for ax,t,letter in zip(axes,ts,'abc'):
        i=int(round(t/.005));ax.imshow(frame(i));ax.axis('off')
        ax.set_title('(%s)  %s'%(letter,{0:'Neutral',4:'Bending',8:'Peak hold'}[t]),loc='left',size=11)
        ax.text(.02,-.035,'%.1f s   |   measured %.1f°'%(t,src['net'][i]),transform=ax.transAxes,color='#315361',size=9)
    fig.subplots_adjust(left=.02,right=.985,top=.85,bottom=.12,wspace=.07)
    fig.savefig(DEFAULT/'cad.png',dpi=210,facecolor='white')
    fig.savefig(DEFAULT/'cad.pdf',facecolor='white')
    plt.close(fig)
    times=src['t']; tcp={n:x['tcp'] for n,x in raw.items()}; angles={n:x['net'] for n,x in raw.items()}
    q=src['qpos']  # Load once for movie playback.
    cmd=src['command']
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf',24)
    ff=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x668','-r','20','-i','-','-an','-c:v','libx264','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(DEFAULT/'integrated.mp4')],stdin=subprocess.PIPE)
    for i in range(0,len(times),10):
        d.qpos[:]=q[i];mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cam,scene_option=opt)
        im=Image.new('RGB',(960,668),'white')
        im.paste(Image.fromarray(renderer.render()),(0,47))
        brush=ImageDraw.Draw(im);brush.text((25,12),'Fixed-base bend response',font=bold,fill='#223a45')
        brush.line((25,49,930,49),fill='#b7c3c7',width=1)
        x=618
        for y,s,f,c in [(97,'SIMULATION TIME',small,'#576973'),(125,'%.2f s'%times[i],bold,'#263d49'),
                        (201,'Issued bending',font,'#263d49'),(232,'%.1f°'%cmd[i],bold,'#263d49'),
                        (302,'Integrated bending',font,'#263d49')]:
            brush.text((x,y),s,font=f,fill=c)
        for n,y in ((8,344),(16,378),(24,412)):
            brush.text((x,y),'%2d elements     %7.3f°'%(n,angles[n][i]),font=font,fill='#176966')
        dev=1000*np.linalg.norm(tcp[8][i]-tcp[24][i])
        brush.text((x,484),'Gripper TCP, 8 vs 24',font=font,fill='#263d49')
        brush.text((x,515),'%.2f mm'%dev,font=bold,fill='#263d49')
        brush.text((x,597),'CAD view: 8-element model',font=small,fill='#576973')
        brush.text((x,624),'Integration step: 0.5 ms',font=small,fill='#576973')
        if i==1600: im.save(DEFAULT/'poster.png')
        ff.stdin.write(im.tobytes())
    ff.stdin.close()
    if ff.wait():raise RuntimeError('video encoding failed')
    renderer.close()
if __name__=='__main__':render()
