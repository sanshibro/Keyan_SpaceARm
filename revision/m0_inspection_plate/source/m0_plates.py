"""Camera-registered illustration, common-view comparisons and measured time histories."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from PIL import Image
from m0_data import (OUT,QA,WORK,Run,SEED,selected,plant,body_poses,unit,write_json,angle,COLORS,LABELS)
from m0_render import HERO,GATE_CAM,DOCK_CAM,Renderer,ghost_picture
from m0_scene import C,A,B,robot_parts

INK="#203a3e"; MUTED="#566d71"; RULE="#c3cecd"; TEAL="#08786a"; COPPER="#ad683d"; RED="#a34e3b"
TIMES=(24.,30.0,35.5,52.)
plt.rcParams.update({"font.family":"Arial","mathtext.fontset":"stix","font.size":9,
    "pdf.fonttype":42,"svg.fonttype":"none","svg.hashsalt":"M0-951c512","image.composite_image":False,
    "axes.spines.right":False,"axes.spines.top":False,"axes.linewidth":.55,
    "axes.edgecolor":"#829598","text.color":INK,"axes.labelcolor":INK,"xtick.color":MUTED,
    "ytick.color":MUTED,"lines.solid_capstyle":"round","figure.facecolor":"white"})
HALO=[pe.Stroke(linewidth=2.3,foreground="white",alpha=.76),pe.Normal()]

def ft(fig,x,y,text,size=9,**kw):
    return fig.text(x,y,text,fontsize=size,**kw)

def headline(fig,letter,title,subtitle):
    ft(fig,.033,.971,letter,17,weight="bold",va="top",color=TEAL)
    ft(fig,.080,.970,title,18,weight="bold",va="top",fontfamily="Times New Roman")
    ft(fig,.082,.922,subtitle,9.1,va="top",color=MUTED)

def drawing(fig,rect,im):
    x,y,w,h=rect
    fw,fh=fig.get_size_inches()
    scale=min(w*fw/im.width,h*fh/im.height)
    new_w,new_h=scale*im.width/fw,scale*im.height/fh
    ax=fig.add_axes((x+(w-new_w)/2,y+(h-new_h)/2,new_w,new_h));ax.set_axis_off()
    ax.imshow(im,interpolation="none",aspect="equal",origin="upper")
    ax.set_xlim(-.5,im.width-.5);ax.set_ylim(im.height-.5,-.5)
    return ax

def stroke(ax,cam,size,pts,color,lw=1.3,ls="-",alpha=1.,halo=True):
    pixels=cam.pixels(pts,size)
    line,=ax.plot(pixels[:,0],pixels[:,1],lw=lw,color=color,ls=ls,alpha=alpha)
    if halo:line.set_path_effects([pe.Stroke(linewidth=lw+1.3,foreground="white",alpha=.77),pe.Normal()])
    return pixels

def xy_note(ax,xy,text,offset,size=8.5,ha="center"):
    ax.annotate(text,xy,xytext=offset,textcoords="offset points",ha=ha,va="center",
                fontsize=size,weight="bold",color=INK,annotation_clip=False,
                bbox=dict(boxstyle="round,pad=.24",fc="white",ec="none",alpha=.92),
                arrowprops=dict(arrowstyle="-",lw=.72,color="#687b7b",shrinkA=2,shrinkB=1.5))

def exported(fig,name,preview=False):
    if preview:
        fig.savefig(QA/"场景审美预览.png",dpi=150,facecolor="white")
    else:
        fig.savefig(OUT/(name+".png"),dpi=300,facecolor="white")
        fig.savefig(WORK/(name+"-layers.svg"),dpi=220,facecolor="white",format="svg")
    plt.close(fig)

def hero(r,f,preview=False):
    size=(1740,920) if preview else (2700,1430)
    if not preview and (WORK/'studio/hero.png').is_file():
        im=Image.open(WORK/'studio/hero.png').convert('RGB');size=im.size
    else:
        im=ghost_picture(r,HERO,f,TIMES,size)
    fig=plt.figure(figsize=(11.6,7.45))
    headline(fig,"a","Through the bulkhead to an oblique inspection port",
             "Recorded whole-body motion   /   pre-entry strain offset   /   representative paired flight 26112131")
    rect=(.035,.145,.930,.747)
    ax=drawing(fig,rect,im)
    phys=f.physics;rot,pos,com=body_poses(r.m,phys["qpos"])
    tcp=phys["tcp"];times=f.pt
    for pts,color in ((com,TEAL),(tcp,COPPER)):
        for start,end,style,width,alpha in ((0,20.4,":",.79,.62),(20.4,28,"-",1.20,.97),
                           (28,32,"-",2.05,1.),(32,52,"-",1.25,.96)):
            sel=(times>=start)&(times<=end+1e-6)
            stroke(ax,HERO,size,pts[sel],color,width,style,alpha)
    # The shutter bracket is a time annotation outside the vehicle envelope.
    cloud=com[(times>=28)&(times<32)]
    brackets=np.array([[cloud[:,0].min(),.46,2.45],[cloud[:,0].min(),.46,2.47],
                       [cloud[:,0].max(),.46,2.47],[cloud[:,0].max(),.46,2.45]])
    uv=stroke(ax,HERO,size,brackets,"#728380",.8,halo=True)
    ax.annotate("Wrist blackout · 28–32 s",uv[1:3].mean(0),xytext=(0,6),textcoords="offset points",ha="center",
                fontsize=8,color=MUTED,path_effects=HALO)
    td=np.linspace(0,2*np.pi,121)
    circle=C+.035*(np.cos(td)[:,None]*B[:,0]+np.sin(td)[:,None]*B[:,1])
    stroke(ax,HERO,size,circle,"#c8824a",.72,ls="--")
    # Bound every mesh vertex on each of the four archived poses, including propellers.
    pieces=robot_parts(r.m);bounds=[]
    for j,stamp in enumerate(TIMES):
        idx=int(np.argmin(abs(times-stamp)))
        cloud=np.vstack([p.v@rot[idx,p.body].T+pos[idx,p.body] for p in pieces])
        uv=HERO.pixels(cloud,size);bb=np.r_[uv.min(0),uv.max(0)]
        if bb[0]<8 or bb[1]<8 or bb[2]>=size[0]-8 or bb[3]>=size[1]-8:
            raise ValueError("Vehicle clipped "+str((stamp,bb)))
        stat=f.row(stamp);th,ph=angle(f.z["true_b"])
        bounds.append(dict(release_s=stamp,pixel_bounds=bb.tolist(),state=str(f.z["phase"][stat]),
                           theta_deg=float(th[stat]),phi_deg=float(ph[stat])))
        root=com[idx];tip=tcp[idx]
        for pt,color in ((root,TEAL),(tip,COPPER)):
            u,v=HERO.pixels(pt,size);ax.scatter(u,v,s=13,facecolor=color,edgecolors="white",linewidths=.65,zorder=6)
        marker=HERO.pixels(root+[0,0,.111],size)
        ax.annotate(str(j+1),marker,xytext=[(-23,16),(13,19),(11,22),(19,16)][j],
                    textcoords="offset points",ha="center",va="center",weight="bold",fontsize=8,
                    color="white",bbox=dict(boxstyle="circle,pad=.25",fc=INK,ec="white",lw=.8),
                    arrowprops=dict(arrowstyle="-",color=INK,lw=.65))
    # Datum scale is in the image plane of the orthographic observer.
    span=.25/HERO.width*size[0];x=.052*size[0];y=.945*size[1]
    ax.plot([x,x+span],[y,y],color=INK,lw=1.55,path_effects=HALO)
    for end in (x,x+span):ax.plot([end,end],[y-5,y+4],color=INK,lw=.85,path_effects=HALO)
    ax.text(x+span*.5,y-10,"250 mm",fontsize=7.2,color=INK,ha="center",va="bottom",path_effects=HALO)
    fig.add_artist(Line2D([.035,.965],[.143,.143],transform=fig.transFigure,lw=.7,color=RULE))
    infos=[("1","Pre-bend after calibration"),("2","Lift during occlusion"),
           ("3","Traverse the sill"),("4","Align to the port")]
    for j,(num,name) in enumerate(infos):
        x=.039+.235*j;ft(fig,x,.106,num,9.5,color=TEAL,weight="bold")
        ft(fig,x+.021,.106,name,9.5,weight="bold")
        ft(fig,x+.021,.073,f"{TIMES[j]:.1f} s",8.4,color=MUTED)
    for x,col,txt in ((.039,TEAL,"Airframe centre of mass"),(.267,COPPER,"Gripper tool centre")):
        fig.add_artist(Line2D([x,x+.036],[.037,.037],transform=fig.transFigure,lw=1.8,color=col))
        ft(fig,x+.043,.031,txt,8,color=MUTED)
    ft(fig,.965,.031,"Aperture 760 × 690 mm    ·    Deck grid 100 mm",8,ha="right",color=MUTED)
    if not preview:
        write_json(QA/"hero-projection.json",dict(camera=HERO.__dict__,size=list(size),records=bounds,
            mesh_parts=len(pieces),arm_nodes=15,gripper_fingers=3,rotor_mesh="visual_super_prop",
            stages=[str(x[1]) for x in infos],state_source="archived world transforms and native CAD"))
    exported(fig,"gate_hero_v2",preview)

def crop_panel(ax,cam,run,i,mode,size):
    z=run.z
    if mode=="gate":
        points=np.stack((z["arm_near_robot"][i],z["arm_near_gate"][i]))
        uv=stroke(ax,cam,size,points,RED,1.5);ax.scatter(*uv.T,s=14,facecolors="white",edgecolors=RED,linewidths=1.)
    else:
        t=np.linspace(0,2*np.pi,121);ring=C+.035*(np.cos(t)[:,None]*B[:,0]+np.sin(t)[:,None]*B[:,1])
        stroke(ax,cam,size,ring,"#b77b42",.8,ls="--")
        ref=stroke(ax,cam,size,np.stack([C-.15*A,C+.035*A]),"#ac753d",1.15,ls="--")
        tip,axis=z["tip"][i],z["tool_axis"][i]
        actual=stroke(ax,cam,size,np.stack([tip-.15*axis,tip+.016*axis]),TEAL,1.34)
        x,y=cam.pixels(C,size);ax.scatter(x,y,s=30,marker="+",color=COPPER,lw=.9)
        x,y=cam.pixels(tip,size);ax.scatter(x,y,s=13,facecolors=TEAL,edgecolors="white",lw=.6)
    ax.set_xlim(-.5,size[0]-.5);ax.set_ylim(size[1]-.5,-.5)

def insets(r,flights):
    fig=plt.figure(figsize=(11.6,7.25))
    headline(fig,"b","Clearance at the sill and alignment at the port",
             "Identical camera, crop and scale within each row   /   same paired seed and optical exposure window")
    ft(fig,.037,.856,"Closest arm–frame approach",13,weight="bold",fontfamily="Times New Roman")
    ft(fig,.965,.855,"Clearance threshold  6.0 cm",8,ha="right",color=RED)
    ft(fig,.037,.459,"Oblique alignment at 52.0 s",13,weight="bold",fontfamily="Times New Roman")
    ft(fig,.965,.459,"Position  ≤ 3.5 cm     ·     Axis  ≤ 3.0°",8,ha="right",color=MUTED)
    size=(1000,600);receipts=[]
    for row,(order,cam,y,mode) in enumerate(((("strain","fixed","proposed"),GATE_CAM,.555,"gate"),
                                           (("strain","oracle","proposed"),DOCK_CAM,.160,"dock"))):
        for col,m in enumerate(order):
            f=flights[m];i=f.closest() if mode=="gate" else f.row(52)
            stamp=f.t[i];asset=WORK/'studio'/(mode+'-'+m+'.png')
            if asset.is_file():
                im=Image.open(asset).convert('RGB');size=im.size
            else:
                r.set_pose(f.pose(stamp));im=r.render(cam,size)
            x=.037+.313*col;ax=drawing(fig,(x,y,.301,.276),im);crop_panel(ax,cam,f,i,mode,size)
            ft(fig,x,y-.030,LABELS[m],9.7,color=COLORS[m],weight="bold")
            if mode=="gate":txt=f"{100*f.z['arm_tool_clearance'][i]:.2f} cm   ·   {stamp:.2f} s"
            else:txt=f"{100*f.z['tip_error'][i]:.2f} cm   /   {f.z['axis_error_deg'][i]:.2f}°"
            ft(fig,x+.301,y-.030,txt,9.1,ha="right",color=INK)
            receipts.append(dict(row=mode,case=f.case,release_s=float(stamp),metric=txt,camera=cam.__dict__))
    ft(fig,.037,.076,"Nearest surfaces",8.5,color=RED,weight="bold")
    ft(fig,.205,.076,"Tool axis",8.5,color=TEAL,weight="bold")
    ft(fig,.324,.076,"Target axis and 35 mm tolerance locus",8.5,color=COPPER)
    v=flights["vision"];stop=np.count_nonzero(v.z["hold_reason"]=="feedback")*np.median(np.diff(v.t))
    ft(fig,.037,.025,f"Vision only held for {stop:.2f} s after wrist feedback expired.  The closest-approach times are specific to each flight.",
       8.7,color=MUTED)
    write_json(QA/"inset-projection.json",receipts)
    exported(fig,"gate_insets_v2")

def curves(flights):
    fig=plt.figure(figsize=(11.6,7.45));headline(fig,"c","Shape feedback during loss and recovery of vision",
        "Four synchronized records   /   archived physical states, feedback and gate geometry   /   seed 26112131")
    handles=[Line2D([],[],color=COLORS[m],lw=(2.4 if m=="proposed" else 1.25),label=LABELS[m]) for m in ("proposed","strain","fixed","vision","oracle")]
    fig.legend(handles=handles,ncol=5,frameon=False,loc="upper left",bbox_to_anchor=(.079,.886),
               borderaxespad=0,handlelength=2.5,fontsize=8.6,columnspacing=1.8)
    gs=fig.add_gridspec(4,1,left=.087,right=.76,bottom=.095,top=.788,hspace=.19)
    axes=[fig.add_subplot(gs[j,0]) for j in range(4)];twin=axes[-1].twinx()
    reference=flights["proposed"];hold=reference.z["blackout"].astype(bool)
    for j in range(2):
        axes[j].plot(reference.t,angle(reference.z["des_b"])[j],color="#263d40",ls=(0,(1.5,2.4)),lw=.93,zorder=7)
    for method in ("vision","oracle","fixed","strain","proposed"):
        f=flights[method];c=COLORS[method];p=method=="proposed";quiet=method=="vision"
        lw=.58 if quiet else (1.87 if p else 1.18);alpha=.47 if quiet else (.69 if method=="oracle" else .96)
        for j in range(2):
            axes[j].plot(f.t,angle(f.z["true_b"])[j],color=c,ls="-",lw=lw,alpha=alpha)
            if method!="oracle":axes[j].plot(f.t,angle(f.z["est_b"])[j],color=c,ls=(0,(3,2)),lw=.67 if quiet else .98,alpha=.35 if quiet else .89)
        axes[2].plot(f.t,100*f.z["body_clearance"],color=c,ls=(0,(1.3,2.)),lw=.55 if quiet else .92,alpha=alpha)
        axes[2].plot(f.t,100*f.z["arm_tool_clearance"],color=c,lw=lw,alpha=alpha)
        crossed=f.z["gate_exit"].astype(bool)
        axes[3].plot(f.t[crossed],100*f.z["tip_error"][crossed],color=c,lw=lw,alpha=alpha)
        twin.plot(f.t[crossed],f.z["axis_error_deg"][crossed],color=c,ls=(0,(3,2)),lw=.57 if quiet else 1.02,alpha=alpha)
    for j in (0,1):
        axes[j].plot(reference.t[hold],angle(reference.z["vision_hold_b"])[j][hold],color="#924b35",lw=1.7,ls="-.",zorder=8)
    for j,(ax,yl,lim) in enumerate(zip(axes,("Bend θ (°)","Azimuth φ (°)","Clearance (cm)","Position (cm)"),
                                     ((21,94),(117,207),(0,32),(0,14)))):
        ax.set_xlim(24,55);ax.set_ylim(*lim);ax.set_ylabel(yl,fontsize=9)
        ax.axvspan(28,32,facecolor="#e6ebe8",zorder=-4)
        ax.grid(axis="y",lw=.45,color="#dfe5e2");ax.tick_params(labelsize=8,length=2.5,pad=3)
        if j<3:ax.tick_params(labelbottom=False)
    for ax,y in ((axes[2],6),(axes[3],3.5)):
        ax.axhline(y,color=RED,lw=.8,ls=(0,(2,2)))
    twin.set_ylim(0,14);twin.tick_params(axis="y",labelsize=7.7,colors=MUTED,length=2);twin.set_ylabel("Axis (°)",fontsize=8.2,color=MUTED,labelpad=0)
    twin.axhline(3,color="#83614a",lw=.68,ls=(0,(5,3)))
    axes[3].set_xlabel("Time after release (s)")
    axes[0].text(30,.95,"4 s occlusion",transform=axes[0].get_xaxis_transform(),ha="center",va="top",fontsize=7.3,color=MUTED)
    x=.832;fig.add_artist(Line2D([x-.017,x-.017],[.108,.791],transform=fig.transFigure,lw=.7,color=RULE))
    def note(y,title,items):
        ft(fig,x,y,title,10.5,weight="bold",fontfamily="Times New Roman")
        for k,(tx,c) in enumerate(items):ft(fig,x,y-(k+1)*.032,tx,8.,color=c)
    fixed=flights['fixed'];strain=flights['strain'];i=strain.closest()
    delta=abs(angle(reference.z['vision_hold_b'])[0][hold]-angle(reference.z['true_b'])[0][hold])
    note(.75,"Dynamic bend",[(f"Last pose peak  {delta.max():.1f}°","#924b35"),
        (f"Fixed KF RMSE  {fixed.metrics['blackout_angle']['theta_rmse_deg']:.2f}°",COLORS['fixed']),
        (f"Proposed RMSE  {reference.metrics['blackout_angle']['theta_rmse_deg']:.2f}°",TEAL)])
    note(.565,"Bend plane",[("Solid   physical state",INK),("Dashed   estimate",MUTED),("Dotted   reference",MUTED),("Dash-dot   last vision","#924b35")])
    note(.395,"Gate surfaces",[("Solid   arm and tool",INK),("Dotted   airframe",MUTED),("Red line   6.0 cm",RED)])
    note(.215,"Docking",[("Solid   position",INK),("Dashed   tool axis",MUTED),("Limits   3.5 cm / 3.0°",RED)])
    exported(fig,"gate_four_metrics_v2")

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args()
    runs=selected();r=Renderer(plant());hero(r,runs['proposed'],args.preview)
    if not args.preview:insets(r,runs);curves(runs)
if __name__=='__main__':main()
