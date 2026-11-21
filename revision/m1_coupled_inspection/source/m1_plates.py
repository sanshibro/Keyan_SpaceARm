"""Publication plates from native renders and paired-flight measurements."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review'
FILM=ROOT/'records/media'
DATA=ROOT/'records/formal'
SEED=26112161
METHODS=('straight','fixed','sequential','unregularized','coupled')
NAMES=('Straight arm','Fixed morphology','Body–then–arm','Joint, no timing cost','Coupled whole-body')
INK='#243936'; MUTED='#61736f'; TEAL='#0b7068'; COPPER='#a95c34'; BLUE='#446e82'; GRAY='#879492'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def setup():
    plt.rcParams.update({'font.family':'serif','font.serif':['STIX Two Text','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':9.5,'axes.labelsize':9.1,'axes.titlesize':10,
        'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':'#84928e',
        'xtick.color':MUTED,'ytick.color':MUTED,'xtick.labelsize':7.9,'ytick.labelsize':7.9,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6,'figure.facecolor':'white'})
def export(f,name):
    OUT.mkdir(exist_ok=True)
    for suffix in ('png','svg','pdf'):
        f.savefig(OUT/(name+'.'+suffix),dpi=265,facecolor='white',pad_inches=0)
    plt.close(f)
def heading(fig,x,y,a,label):
    fig.text(x,y,a,fontsize=12.5,weight='bold',va='top')
    fig.text(x+.025,y,label,fontsize=12.1,va='top')
def hero():
    p=read(FILM/'projection.json')
    f=plt.figure(figsize=(10.5,4.83))
    a=f.add_axes([.008,.008,.984,.933]);a.imshow(Image.open(FILM/'hero-native.png'),extent=[0,1440,810,0])
    a.set_aspect('equal',adjustable='box');a.set_xlim(8,1431);a.set_ylim(735,49);a.axis('off')
    for key,color,style in [('body',TEAL,'-'),('tool',COPPER,'--')]:
        paths=np.array(p[key]);a.plot(*paths.T,color='white',lw=2.95,alpha=.84,zorder=2)
        a.plot(*paths.T,color=color,ls=style,lw=1.55,zorder=3)
    for i,pose in enumerate(p['poses']):
        x,y=pose['centre']
        # Place time markers below the base trajectory, leaving the propellers fully visible.
        dx,dy=[(-26,68),(2,70),(8,76),(36,62)][i]
        a.annotate('%.1f s'%pose['t'],(x,y+10),xytext=(x+dx,y+dy),ha='center',va='center',fontsize=8,
             color=INK,bbox=dict(facecolor='white',edgecolor='none',alpha=.92,pad=1.6),
             arrowprops=dict(arrowstyle='-',color='#586e68',lw=.65,shrinkA=1,shrinkB=1))
    for xy,offset,label in zip(p['landmarks'],[(-80,-105),(96,-86),(94,56)],['Low beam','Offset riser','Inspection flange']):
        x,y=xy;dx,dy=offset
        a.annotate(label,(x,y),xytext=(x+dx,y+dy),ha='center',va='center',fontsize=8.7,
              bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=1.7),
              arrowprops=dict(arrowstyle='-',lw=.6,color=MUTED,shrinkA=1,shrinkB=2))
    heading(f,.023,.975,'a','A whole-body trajectory through the service gallery')
    leg=[Line2D([0],[0],color=TEAL,lw=1.8),Line2D([0],[0],color=COPPER,ls='--',lw=1.8)]
    f.legend(leg,['Aircraft frame origin','Tool centre'],loc='upper right',bbox_to_anchor=(.98,.987),ncol=2,frameon=False,fontsize=8.4)
    export(f,'m1_hero')
def insets():
    sec=read(FILM/'sections.json');cuts=read(FILM/'crosssections.json');f=plt.figure(figsize=(10.5,6.15))
    cols=[.023,.354,.684];w=.299
    for i,(tag,title,base) in enumerate([('beam','Low service opening','straight'),('riser','Offset pipe & instrument tap','fixed'),('flange','Oblique inspection flange','sequential')]):
        x=cols[i];heading(f,x,.984,'bcd'[i],title)
        for y,m in ((.648,base),(.356,'coupled')):
            ax=f.add_axes([x,y,w,.233]);im=Image.open(FILM/(tag+'-'+m+'.jpg'))
            # Same view, scale and crop for both members of each pair.
            ax.imshow(im);ax.axis('off')
            r=sec[tag][m];title=NAMES[METHODS.index(m)]
            f.text(x,y+.233,title,fontsize=8.6,color=TEAL if m=='coupled' else INK,va='bottom')
            f.text(x+w,y+.233,'x = %.2f m'%r['x'],fontsize=8,color=MUTED,ha='right',va='bottom')
        ax=f.add_axes([x+.014,.066,w-.020,.222]);ax.set_aspect('equal')
        z=cuts[tag]['coupled']
        for group,c in [('walls','#d9dfda'),('body',BLUE),('arm',TEAL)]:
            for poly in z['parts'][group]:
                pts=np.array(poly);ax.fill(*pts.T,facecolor=c,edgecolor='#a0aaa5' if group=='walls' else c,lw=.3)
        for group in ('body','arm'):
            for poly in cuts[tag][base]['parts'][group]:
                pts=np.array(poly+[poly[0]]);ax.plot(*pts.T,color=COPPER,lw=.8,ls='--')
        ax.set_xlim(-.65,.75);ax.set_ylim(1.35,2.53);ax.set_xticks([-.5,0,.5]);ax.set_yticks([1.5,2.0,2.5])
        ax.tick_params(length=2,pad=1,labelsize=6);ax.spines[['top','right']].set_visible(False)
        ax.set_xlabel('Transverse y (m)',fontsize=7,labelpad=.6)
        if i==0:ax.set_ylabel('Height z (m)',fontsize=7,labelpad=.5)
        f.text(x,.299,'Convex section at x = %.2f m'%z['plane'],fontsize=8.5,color=MUTED)
    export(f,'m1_sections')
def curve(): 
    f,axes=plt.subplots(4,2,figsize=(10.5,7.55),sharex=True)
    f.subplots_adjust(left=.074,right=.947,top=.944,bottom=.071,hspace=.49,wspace=.24)
    axs=axes.ravel()
    for a in axs:
        a.spines[['top','right']].set_visible(False);a.grid(axis='y',color='#dce3df',lw=.55)
        a.tick_params(length=2.3,width=.6);a.margins(x=.013)
    def plot(ax,t,y,col=TEAL,dash='-',width=1.2,z=3,alpha=1):
        ax.plot(t,y,color=col,ls=dash,lw=width,zorder=z,alpha=alpha)
    S={}
    for m in ('coupled','unregularized'):
        with np.load(DATA/(m+'-'+str(SEED))/'audit/series.npz') as v:S[m]={k:v[k] for k in v.files}
    o=S['coupled']; ix=o['t']>=21; t=o['t'][ix]-21
    full=DATA/('coupled-'+str(SEED))
    with np.load(full/'flight/control.npz') as c,np.load(full/'plan/certified.npz') as q:
        assert np.allclose(c['t'],o['t'])
        clock=c['spline_t'][ix];sched=np.array([np.interp(clock,q['t'],col) for col in q['q'].T]).T
    for i,col in enumerate([TEAL,COPPER,BLUE]):plot(axs[0],t,o['body'][ix,i],col)
    planned='#87674f'
    for i in range(3):plot(axs[0],t,sched[:,i],planned,':',.85,4,.72)
    plot(axs[1],t,np.rad2deg(sched[:,3]),planned,':',.95,4)
    plot(axs[2],t,np.rad2deg(sched[:,4]),planned,':',1.05,4)
    phi_ref=np.rad2deg(np.unwrap(sched[:,5]));phi_ref[np.rad2deg(sched[:,4])<8]=np.nan
    if np.any(np.isfinite(phi_ref)):
        phi_ref+=360*round((180.-np.nanmedian(phi_ref))/360);plot(axs[3],t,phi_ref,planned,':',1.05,4)
    for m,col,sty in [('unregularized',GRAY,'--'),('coupled',TEAL,'-')]:
        z=S[m];k=z['t']>=21;tm=z['t'][k]-21;b=z['actual_b'][k];theta=np.rad2deg(np.linalg.norm(b,axis=1))
        phi=np.rad2deg(np.unwrap(np.arctan2(b[:,1],b[:,0])));valid=theta>=8.
        if valid.any():phi-=360*np.round(np.median(phi[valid])/360)
        phi[~valid]=np.nan
        plot(axs[1],tm,np.rad2deg(np.unwrap(z['yaw'][k])),col,sty)
        plot(axs[2],tm,theta,col,sty);plot(axs[3],tm,phi,col,sty)
        plot(axs[4],tm,np.linalg.norm(z['mount_reaction'][k,:2],axis=1),col,sty)
        plot(axs[5],tm,z['front_rear'][k],col,sty)
        plot(axs[6],tm,100*z['tracking'][k],col,sty)
    plot(axs[4],t,np.linalg.norm(o['moment'][ix],axis=1),COPPER,':',1.1)
    plot(axs[7],o['self_exact_t']-21,100*o['external_exact'][:,0],BLUE)
    plot(axs[7],o['self_exact_t']-21,100*o['external_exact'][:,1],COPPER)
    self_ax=axs[7].twinx();self_ax.spines['top'].set_visible(False);self_ax.spines['left'].set_visible(False)
    plot(self_ax,o['self_exact_t']-21,1000*o['self_exact'],TEAL,':',1.0,alpha=.76)
    self_ax.set_ylim(0,max(2.4,1000*np.max(o['self_exact'])*1.45));self_ax.set_yticks([0,1,2])
    self_ax.set_ylabel('Self gap (mm)',fontsize=7,color=TEAL,labelpad=1)
    self_ax.tick_params(axis='y',labelsize=6.8,length=2,pad=1,colors=TEAL)
    axs[7].axhline(6,color='#874d45',ls=(0,(2,3)),lw=.8)
    axs[7].annotate('6 cm',xy=(t[-1],6),xytext=(-1,2),textcoords='offset points',ha='right',fontsize=6.5,color='#874d45')
    axs[7].set_ylim(-1, min(53,max(30,100*np.nanmax(o['external_exact'])+1)))
    captions=[('Base translation','Position (m)'),('Base heading','Yaw (deg)'),('Bending magnitude','θ (deg)'),
        ('Bending plane','φ (deg)'),('Horizontal moment at the mount','Moment (N m)'),
        ('Front–rear rotor loading','Thrust difference (N)'),('Position tracking','3-D error (cm)'),
        ('Body, tool and self-separation','Distance (cm)')]
    for i,(ttl,ylabel) in enumerate(captions):
        axs[i].set_title(chr(97+i)+'   '+ttl,loc='left',fontweight='bold',pad=4,fontsize=9.3)
        axs[i].set_ylabel(ylabel,labelpad=3)
    event_ids=[np.flatnonzero(ix&(o['body'][:,0]>=x)) for x in (1.,1.77)]
    event_ids+=[np.flatnonzero(ix&(o['phase']=='inspect'))]
    for ids,name in zip(event_ids,['Beam','Riser','Inspect']):
        if not len(ids):continue
        tm=o['t'][ids[0]]-21
        for a in axs:a.axvline(tm,color='#8e9e97',lw=.55,ls=(0,(1,3)),zorder=1,alpha=.72)
        axs[6].text(tm+.55,.97,name,transform=axs[6].get_xaxis_transform(),ha='left',va='top',fontsize=6.1,color=MUTED)
    for a in axs[-2:]:a.set_xlabel('Time from planned motion onset (s)')
    for a,entries,cl,styles in [(axs[0],['x','y','z'],[TEAL,COPPER,BLUE],['-']*3),
        (axs[4],['Gravity, coupled plan'],[COPPER],[':']),
        (axs[7],['Body','Arm/tool','Self (right)'],[BLUE,COPPER,TEAL],['-','-',':'])]:
        a.legend([Line2D([0],[0],color=c,ls=s,lw=1.25) for c,s in zip(cl,styles)],entries,
                 loc='upper right',frameon=False,fontsize=6.7,ncol=len(entries),columnspacing=.8,handlelength=1.4)
    f.legend([Line2D([0],[0],color=TEAL,lw=1.7),Line2D([0],[0],color=planned,ls=':',lw=1.1),Line2D([0],[0],color=GRAY,ls='--',lw=1.4)],
        ['Coupled flight','Coupled spline (a–d)','Joint flight without coupling-timing penalties'],loc='upper center',bbox_to_anchor=(.5,1.0),ncol=3,frameon=False,fontsize=8.7)
    export(f,'m1_curves')
if __name__=='__main__':
    setup();hero();insets();curve()
