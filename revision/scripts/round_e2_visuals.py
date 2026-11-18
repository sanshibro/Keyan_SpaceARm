"""Make the E2 review sheet and replay from the recorded time-aligned solver states."""
import argparse,csv,json,subprocess
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from round_e2_run import DEFAULT,build,neutral,tag

INK='#213745';GREY='#556a74';TEAL='#1e706c';COPPER='#ad5937';BLUE='#516d91'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42,'axes.spines.right':False,
    'axes.spines.top':False,'text.color':INK,'axes.edgecolor':'#71838a','axes.labelcolor':INK,
    'xtick.color':INK,'ytick.color':INK,'axes.linewidth':.65,'savefig.facecolor':'white'})
def paths(out):
    arrays=[]
    for g in (0,100):
        with np.load(str(out/(tag(8,.0005,1,1,g)+'.npz'))) as x:arrays.append({k:x[k] for k in ('t','state_t','qpos','net','tcp','command')})
    return arrays

def render(out,arrays):
    m,_=build(8,.0005);d=neutral(m);w=h=440;r=mujoco.Renderer(m,height=h,width=w)
    cam=mujoco.MjvCamera();cam.orthographic=1;cam.lookat[:]=[.045,0,1.28];cam.distance=1.1;cam.azimuth=108;cam.elevation=-9
    m.vis.global_.fovy=.81
    opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[3]=0;opt.sitegroup[:]=0
    def pose(x,k,color):
        d.qpos[:]=x['qpos'][k];mujoco.mj_forward(m,d)
        r.update_scene(d,camera=cam,scene_option=opt)
        im=Image.fromarray(r.render().copy()).convert('RGB')
        cam1,cam2=r.scene.camera
        fwd=np.array(cam1.forward);up=np.array(cam1.up);right=np.cross(fwd,up)
        origin=(np.asarray(cam1.pos)+np.asarray(cam2.pos))/2
        rel=d.site('tip').xpos-origin;span=cam1.frustum_top-cam1.frustum_bottom
        px=w/2+h*np.dot(rel,right)/span;py=h*(.5-np.dot(rel,up)/span)
        brush=ImageDraw.Draw(im);brush.ellipse((px-4,py-4,px+4,py+4),outline='white',width=3)
        brush.ellipse((px-3,py-3,px+3,py+3),outline=color,width=2)
        return im
    for x,g,col in zip(arrays,(0,100),(TEAL,COPPER)):
        pose(x,int(np.argmin(abs(x['state_t']-8.8))),col).save(out/('cad_%dg.png'%g))
    regular='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';serif='/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf'
    ft=ImageFont.truetype(regular,18);sm=ImageFont.truetype(regular,15);big=ImageFont.truetype(serif,25)
    ff=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x590','-r','25','-i','-','-an','-c:v','libx264','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'replay.mp4')],stdin=subprocess.PIPE)
    for frame in range(350):
        k=frame*2;j=k*4;t=arrays[0]['t'][j]
        canvas=Image.new('RGB',(960,590),'white');draw=ImageDraw.Draw(canvas)
        draw.text((25,12),'Fixed-base response to added tool load',font=big,fill=INK)
        draw.text((780,18),'t = %5.2f s'%t,font=ft,fill=GREY)
        draw.line((24,57,936,57),fill='#b6c3c8')
        for i,(x,g,col) in enumerate(zip(arrays,(0,100),(TEAL,COPPER))):
            left=24+i*476;canvas.paste(pose(x,k,col),(left,85))
            draw.text((left,66),'Added load   %d g'%g,font=ft,fill=col)
            draw.text((left,510),'Integrated bend   %.2f°'%x['net'][j],font=ft,fill=INK)
        sep=1000*np.linalg.norm(arrays[0]['tcp'][j]-arrays[1]['tcp'][j])
        draw.line((24,544,936,544),fill='#b6c3c8')
        draw.text((25,554),'Issued %.1f°      |      3-D TCP separation %.2f mm'%(arrays[0]['command'][j],sep),font=ft,fill=INK)
        draw.text((650,557),'Rings mark the gripper tool center.',font=sm,fill=GREY)
        if frame==220:canvas.save(out/'poster.png')
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    if ff.wait():raise RuntimeError('video encoder failed')
    r.close()

def cad_panels(fig,rect,arrays,out):
    x0,y0,w,h=rect
    for i,(x,g,col) in enumerate(zip(arrays,(0,100),(TEAL,COPPER))):
        ax=fig.add_axes([x0+i*w*.316,y0,w*.265,h])
        ax.imshow(plt.imread(str(out/('cad_%dg.png'%g))));ax.axis('off')
        j=int(np.argmin(abs(x['t']-8.8)))
        ax.set_title('(%s)  %d g added'%('ab'[i],g),fontsize=10,loc='left')
        ax.text(0,-.075,'Bend  %.2f°'%x['net'][j],transform=ax.transAxes,color=col,fontsize=9)
    j=int(np.argmin(abs(arrays[0]['t']-8.8)))
    d=1000*(arrays[1]['tcp'][j]-arrays[0]['tcp'][j])
    ax=fig.add_axes([x0+w*.716,y0+.08*h,w*.246,h*.80])
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlim(-5,5);ax.set_xticks([-4,0,4]);ax.set_ylim(min(-8.,d[2]-1.5),2)
    ax.plot([0,d[0]],[0,d[2]],lw=1.1,color='#87999e')
    ax.scatter([0,d[0]],[0,d[2]],s=26,color=[TEAL,COPPER],zorder=3)
    ax.annotate('0 g',(0,0),xytext=(6,3),textcoords='offset points',color=TEAL,fontsize=8)
    ax.annotate('100 g',(d[0],d[2]),xytext=(5,-1),textcoords='offset points',color=COPPER,fontsize=8)
    ax.grid(color='#e3e8e9',linewidth=.55);ax.tick_params(labelsize=7,length=3)
    ax.set_title('(c)  TCP in the x–z plane',fontsize=10,loc='left')
    ax.set_xlabel('Δx (mm)',fontsize=8);ax.set_ylabel('Δz (mm)',fontsize=8)

def paired_curves(fig,axes,rows,letters):
    dat={(float(r['k_scale']),float(r['d_scale']),int(r['load_g'])):r for r in rows if int(r['N'])==8 and float(r['dt_ms'])==.5}
    a,b=axes
    for K,colour,marker in [(.75,COPPER,'s'),(1.,TEAL,'o'),(1.25,BLUE,'^')]:
        a.plot([0,50,100],[float(dat[K,1.,g]['hold_deficit_deg']) for g in [0,50,100]],label='%.2f×'%K,
               color=colour,marker=marker,lw=1.5,ms=4)
    for g,colour,marker in [(0,TEAL,'o'),(50,BLUE,'^'),(100,COPPER,'s')]:
        b.plot([.6,1,1.4],[float(dat[1.,d,g]['rise_delay_ms']) for d in [.6,1,1.4]],label='%d g'%g,color=colour,marker=marker,lw=1.5,ms=4)
    a.set_xlim(-6,110);a.set_xticks([0,50,100]);a.set_ylim(1.4,5.6);a.set_yticks([2,3,4,5])
    b.set_xlim(.54,1.48);b.set_xticks([.6,1.,1.4]);b.set_ylim(70,135);b.set_yticks([75,90,105,120,135])
    a.set_title('(%s)  Holding under load'%letters[0],loc='left',fontsize=10)
    b.set_title('(%s)  Transient response'%letters[1],loc='left',fontsize=10)
    a.set_xlabel('Added mass (g)');a.set_ylabel('Hold deficit (deg)')
    b.set_xlabel('Drive damping / nominal');b.set_ylabel('Rise delay (ms)')
    for ax,title in [(a,'Stiffness'),(b,'Load')]:
        ax.grid(axis='y',color='#dce4e5',lw=.55);ax.set_axisbelow(True);ax.tick_params(labelsize=8,length=3)
        ax.legend(title=title,loc='upper left',borderaxespad=.4,frameon=False,ncol=1,fontsize=7,title_fontsize=7,handlelength=1.6,labelspacing=.18)

def figures(out,arrays):
    summary=json.loads((out/'summary.json').read_text());rows=list(csv.DictReader((out/'matrix.csv').open()))
    f=plt.figure(figsize=(8,3));cad_panels(f,[.035,.215,.92,.6],arrays,out);f.savefig(out/'cad.png',dpi=200);plt.close(f)
    f,axes=plt.subplots(1,2,figsize=(7.7,2.75));f.subplots_adjust(left=.082,right=.98,bottom=.21,top=.88,wspace=.37)
    paired_curves(f,axes,rows,'ab');f.savefig(out/'curves.png',dpi=200);plt.close(f)
    f=plt.figure(figsize=(8.27,10.6))
    f.text(.074,.952,'Sensitivity of the continuum arm to load and actuation',fontsize=15.5,weight='bold')
    f.text(.075,.925,'Fixed SUPER airframe  /  Qingyu arm  /  27 parameter combinations and 3 numerical controls',fontsize=9,color=GREY)
    f.add_artist(Line2D([.075,.927],[.910,.910],transform=f.transFigure,color='#9aabae',lw=.65))
    f.text(.075,.878,'Where does the tool settle under the same 118° command?',fontsize=11,style='italic')
    f.text(.075,.844,'Two integrated states at 8.80 s share the same geometry and servo settings. Colored rings mark the tool\n'
                       'center point (TCP). The inset resolves the displacement from the unloaded hold on metric axes.',fontsize=9,linespacing=1.45)
    cad_panels(f,[.076,.623,.84,.190],arrays,out)
    axes=[f.add_axes([.103,.360,.342,.196]),f.add_axes([.573,.360,.342,.196])]
    paired_curves(f,axes,rows,'de')
    f.text(.075,.291,'One-factor contrasts at the nominal settings of the other two factors',fontsize=10.5,weight='bold')
    cols=[.077,.48,.691,.921]
    for j,h in enumerate(['Parameter interval','Δ hold deficit (deg)','Δ rise delay (ms)','Max. TCP gap (mm)']):
        f.text(cols[j],.260,h,ha='left' if j==0 else 'right',size=8,color=GREY)
    for i,(key,label) in enumerate([('added_payload','Added mass     0 → 100 g'),('stiffness','Stiffness           0.75 → 1.25×'),('drive_damping','Drive damping  0.60 → 1.40×')]):
        v=summary['anchored_contrasts'][key];h=v['delta_hold_deficit_deg']
        vals=[label,('%+.2f'%h) if abs(h)>=.005 else '<0.01', '%+.2f'%v['delta_rise_delay_ms'],'%.2f'%v['max_paired_tcp_mm']]
        for j,val in enumerate(vals):
            f.text(cols[j],.232-.026*i,val,ha='left' if j==0 else 'right',fontsize=9)
    for y in (.251,.163):f.add_artist(Line2D([.075,.925],[y,y],transform=f.transFigure,color='#b6c3c6',lw=.65))
    f.text(.075,.106,'Hold deficit is command minus mean bend from 8.5 to 9.0 s. Rise delay is the time offset at the\n'
                       '59° rising crossing. TCP gaps are maxima over the full 14 s trajectory. Added mass is represented by\n'
                       'an inertial load at the TCP. The tested intervals are numerical perturbations.',fontsize=8.4,linespacing=1.45)
    ck=summary['numerical_controls']
    f.text(.075,.052,'Largest TCP difference in two 8-to-24 checks: %.3f mm. Time-step halving: %.4f mm.'%(
               max(c['tcp_mm'] for c in ck if c['test']=='coarse_fine'),next(c['tcp_mm'] for c in ck if c['test']=='time_halving')),fontsize=8,color=GREY)
    f.savefig(out/'review.pdf');plt.close(f)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--figures-only',action='store_true');args=p.parse_args()
    data=paths(args.out)
    if not args.figures_only:render(args.out,data)
    figures(args.out,data)
