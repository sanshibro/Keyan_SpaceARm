"""Wrist camera pixels, sensor confidence and shape errors on the actual replay clock."""
import subprocess
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from round_patent_camera import OUT
from round_shape_pixels import OUTPUT
from round_wrist_fusion import readj,CASES
from patent_confidence import angles
BLUE='#245b7a';GREEN='#167466';RED='#a75035';GRAY='#5f6d75';INK='#25373f';LIGHT='#e4e9e8'
CASE='241119-online_pause';RD=OUT/'review';RD.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.3,'axes.spines.top':False,'axes.spines.right':False,
   'axes.edgecolor':'#a2afaf','axes.linewidth':.7,'xtick.color':GRAY,'ytick.color':GRAY,'text.color':INK,
   'axes.labelcolor':INK,'axes.titleweight':'semibold','legend.frameon':False})
def files(fig,stem):
    name=RD/(stem+'.png');fig.savefig(name,dpi=185,facecolor='white')
    with Image.open(name) as im:im.convert('RGB').save(RD/(stem+'.webp'),'WEBP',quality=88,method=6)
    plt.close(fig)
def dash(ax):
    ax.grid(axis='y',lw=.55,color=LIGHT);ax.set_axisbelow(True)
def figures():
    z=np.load(OUT/'traces'/CASE/'zero_offset-graded.npz')
    truth=np.load(OUTPUT/'runs'/CASE/'scoring_truth.npz')['mocap_b'];t=z['t']
    s=np.load(OUT/'camera'/CASE/'samples.npz'); v=np.load(OUT/'support'/(CASE+'-graded.npz'))
    fig=plt.figure(figsize=(8.8,5.35));gs=fig.add_gridspec(3,4,left=.102,right=.975,bottom=.092,top=.925,
        hspace=.70,wspace=.12,height_ratios=[1.08,.88,1.03])
    for j,(im,ts,title) in enumerate(zip(s['rgb'],s['t'],['Clear','Partial mask','Full mask','Recovered'])):
        ax=fig.add_subplot(gs[0,j]);ax.imshow(im);ax.set_xticks([]);ax.set_yticks([])
        for k in ax.spines:ax.spines[k].set_visible(True);ax.spines[k].set_color('#b5bcbd')
        ax.set_title('%s  ·  %.2f s'%(title,ts),loc='left',fontsize=8,pad=3)
    axs=[fig.add_subplot(gs[k,:]) for k in (1,2)]
    for ax in axs:
        ax.axvspan(26.2,31.2,color='#e9e6da',alpha=.65);ax.axvspan(31.2,31.52,color='#829095',alpha=.45)
        ax.set_xlim(20,39);dash(ax)
    ax=axs[0];ax.plot(v['t']+.2,v['cv0'],color=BLUE,lw=.75,label='Tracking statistics')
    ax.plot(v['t']+.2,v['cv'],color=RED,lw=.9,label='Including image coverage (exploratory)')
    ax.set_ylim(-.04,1.07);ax.set_ylabel('Visual confidence');ax.set_title('a   What the tracker can still see',loc='left',pad=4,fontsize=10)
    ax.legend(loc='upper right',fontsize=7.2,ncol=1,borderaxespad=.28)
    ax=axs[1];ref=angles(truth)[:,0]
    for i,col,lbl in [(2,GRAY,'Previous fusion'),(3,BLUE,'Adaptive'),(4,GREEN,'Adaptive + bias correction')]:
        ax.plot(t,angles(z['b'][i])[:,0]-ref,color=col,lw=1.25 if i==4 else .9,
            ls='--' if i==2 else '-',label=lbl)
    ax.axhline(0,color='#838e91',lw=.6);ax.set_ylabel('Bend error (deg)');ax.set_xlabel('Time at estimator output (s)')
    ax.set_title('b   Bend estimate minus virtual motion capture, with strain zero offset',loc='left',pad=4,fontsize=10)
    ax.legend(loc='lower left',fontsize=7.2,ncol=1,borderaxespad=.3)
    files(fig,'tracking')
    scores=readj(OUT/'paired-scores.json')
    fig,axs=plt.subplots(1,2,figsize=(8.8,2.80),sharey=True,gridspec_kw={'width_ratios':[1,1]})
    fig.subplots_adjust(left=.19,right=.973,wspace=.18,bottom=.205,top=.75)
    methods=[('previous','Previous fusion',GRAY,'s'),('adaptive','Adaptive',BLUE,'o'),('corrected','Adaptive + correction',GREEN,'D')]
    for ax,opt,phase,title in zip(axs,['natural','occluded'],['all','same_four_seconds'],['a   Clear camera  ·  32 s','b   Full visual loss  ·  4 s']):
        ax.set_title(title,loc='left',pad=7,fontsize=10)
        for y,g in enumerate(['nominal','zero_offset','material_memory']):
            r=scores[g][opt][phase];x=[r[m]['all']['theta_rmse_deg'] for m,_,_,_ in methods]
            ax.plot([min(x),max(x)],[y,y],lw=1,color='#aab5b5')
            for dy,(m,lab,col,mk),xi in zip([-.13,0,.13],methods,x):
                ax.scatter(xi,y+dy,c=col,s=26,marker=mk,label=lab if y==0 else None,zorder=3)
        ax.set_xlim(-.18,8.6);ax.set_xticks([0,2,4,6,8]);ax.set_xlabel('Bend RMSE (deg)')
        ax.xaxis.grid(True,color=LIGHT,lw=.6);ax.set_axisbelow(True)
    axs[0].set_yticks([0,1,2]);axs[0].set_yticklabels(['Nominal strain','Zero offset','Material memory'])
    axs[0].set_ylim(2.54,-.54);fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,fontsize=8)
    files(fig,'paired')
def movie():
    from matplotlib.font_manager import findfont
    regular=findfont('DejaVu Sans');bold=findfont('DejaVu Sans:weight=bold')
    font=lambda sz,heavy=False:ImageFont.truetype(bold if heavy else regular,sz)
    head=font(15,True);small=font(12);big=font(20)
    z=np.load(OUT/'traces'/CASE/'zero_offset-graded.npz');sup=np.load(OUT/'support'/(CASE+'-graded.npz'))
    truth=angles(np.load(OUTPUT/'runs'/CASE/'scoring_truth.npz')['mocap_b'])[:,0];t=z['t'];bend=angles(z['b'][4])[:,0]
    src=OUT/'inputs'/CASE;rows=[x.split() for x in (src/'frames.txt').read_text().splitlines()]
    W,H=640,316;enc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}',
        '-r','5','-i','-','-an','-c:v','libx264','-preset','slow','-crf','30','-pix_fmt','yuv420p','-movflags','+faststart',str(RD/'replay.mp4')],stdin=subprocess.PIPE)
    timeline=np.arange(21.0,38.001,.20)
    for output in timeline:
        idx=int(np.searchsorted(t,output-1e-7));k=np.searchsorted(sup['t']+.2,output+1e-7,side='right')-1
        im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im)
        d.text((13,8),'Wrist RGB-D and strain during a partial occlusion',font=head,fill=INK)
        stamp,path,_=rows[k];rgb=Image.open(src/path).convert('RGB').resize((310,232),Image.Resampling.LANCZOS if hasattr(Image,'Resampling') else Image.LANCZOS)
        im.paste(rgb,(12,39));d.rectangle((11,38,322,272),outline='#b2bebd')
        d.text((12,281),f'Exposure   {float(stamp):05.2f} s',font=small,fill=GRAY)
        d.text((192,281),f'Output   {output:05.2f} s',font=small,fill=GRAY)
        d.text((340,40),'Bend estimate',font=small,fill=GRAY);d.text((339,56),f'{bend[idx]:5.1f} deg',font=big,fill=GREEN)
        d.text((476,40),'Virtual mocap',font=small,fill=GRAY);d.text((475,56),f'{truth[idx]:5.1f} deg',font=big,fill=INK)
        for y,label,value,color in [(106,'Tracking confidence',sup['cv0'][k],BLUE),(159,'With image coverage',sup['cv'][k],RED)]:
            d.text((340,y),label,font=small,fill=INK);d.text((581,y),f'{value:.2f}',font=small,fill=color)
            d.rectangle((340,y+23,620,y+31),fill='#e2e8e7')
            if value>0:d.rectangle((340,y+23,int(340+280*value),y+31),fill=color)
        d.text((340,219),'Observation at exposure',font=small,fill=GRAY)
        d.text((340,241),'Registered wrist pose' if sup['cv0'][k]>0 else 'Vision unavailable',font=head,fill=GREEN if sup['cv0'][k]>0 else RED)
        d.text((340,283),'Same recorded motion  /  simulated zero offset',font=small,fill=GRAY)
        enc.stdin.write(im.tobytes())
    enc.stdin.close()
    if enc.wait()!=0:raise RuntimeError('FFmpeg failed')
if __name__=='__main__':figures();movie()
