"""Turn retained files into meeting media. No simulator or hardware is launched."""
import os, sys, io, json, tempfile, subprocess as sp, math, bisect, tarfile, csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--repo',required=True);repo=str(Path(ap.parse_args().repo).resolve()); a='688e16adc3f0e33bf49f97c16a5abb2516a2f137'; c='1327b59aa67e2b1440659b5fa4fb6515e859e2b0'
def blob(rev,p):return sp.check_output(['git','-C',repo,'show',rev+':'+p]);
with tempfile.TemporaryDirectory(prefix='meeting-round-') as tmp:
    root=Path(tmp); A=root/'a'; C=root/'c1'; A.mkdir();C.mkdir(); ft='/usr/share/fonts/truetype/dejavu/'
    font={n:ImageFont.truetype(ft+'DejaVuSans.ttf',n) for n in (14,16,18,22)}; serif=ImageFont.truetype(ft+'DejaVuSerif.ttf',27)
    r=json.loads(blob(a,'revision/live/replay.json')); frames=r['frames']; tt=[f['t'] for f in frames]; gates=r['gates']; W,H=1104,560
    base=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(base); ink='#172c39'; blue='#235d7a'; orange='#b66337'; rule='#dce3e6'; green='#20776d'
    def label(d,x,y,text,n=16,co=ink):d.text((x,y),text,font=font[n],fill=co)
    label(d,32,17,'A  /  ARCHIVED SIMULATION LOG',14,blue);d.text((32,42),'Three passages: recorded vehicle and arm geometry',font=serif,fill=ink)
    sx=lambda s:80+int(64*s); zy=lambda z:368-int((z-.40)*141)
    for z in [.5,1,1.5,2]:
        y=zy(z);d.line((80,y,726,y),fill=rule);label(d,42,y-9,str(z),14)
    for s in range(0,11,2):label(d,sx(s)-4,373,str(s),14)
    label(d,82,94,'World height (m)',16);label(d,500,393,'Forward progress (m)',14)
    for g in gates:
        x=sx(g['s']);d.line((x,zy(2.1),x,zy(g['upper'])),fill='#9aa9ae',width=5);d.line((x,zy(g['lower']),x,zy(.4)),fill='#9aa9ae',width=5);label(d,x-11,91,g['name'],14)
    trail=[(sx(f['s']),zy(f['z'])) for f in frames];d.line(trail,fill='#aebdc4',width=2)
    d.line((755,96,755,410),fill=rule);label(d,780,94,'Gate-local section',18)
    xt=lambda t:80+int((t-tt[0])/(tt[-1]-tt[0])*966);ya=lambda angle:508-int(angle*.50)
    for g in gates:
        f0=min(frames,key=lambda f:abs(f['s']-g['s']));x=xt(f0['t']);d.line((x,438,x,508),fill=rule);label(d,x-9,513,g['name'],14)
    d.line([(xt(f['t']),ya(f['theta'])) for f in frames],fill=blue,width=2);label(d,82,412,'Recorded bending angle (deg)',14);label(d,854,513,'Recorded time (s)',14);label(d,35,436,'120',14);label(d,56,496,'0',14)
    rate=15;speed=4.;nf=math.ceil((tt[-1]-tt[0])/speed*rate)+1;movie=A/'replay.mp4'
    ff=sp.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(rate),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)],stdin=sp.PIPE)
    for i in range(nf):
        t=min(tt[-1],tt[0]+i/rate*speed); j=min(bisect.bisect_left(tt,t),len(tt)-1)
        if j and abs(tt[j-1]-t) < abs(tt[j]-t):j-=1
        f=frames[j]; im=base.copy();dr=ImageDraw.Draw(im);s,z=f['s'],f['z'];g=min(gates,key=lambda q:abs(q['s']-s)); px,py=sx(s),zy(z);dr.line([(sx(p[0]),zy(p[1])) for p in f['arm']],fill=green,width=4);dr.ellipse((px-4,py-4,px+4,py+4),fill=blue)
        label(dr,790,44,'4x replay  |  t = %5.1f s'%f['t'],18,blue);label(dr,780,135,g['name']+'  /  side projection',16);gx=924; zz=lambda z:360-int((z-.6)*149);gg=lambda s:gx+int((s-g['s'])*149)
        dr.line((gx,zz(1.98),gx,zz(g['upper'])),fill='#9aa9ae',width=5);dr.line((gx,zz(g['lower']),gx,zz(.6)),fill='#9aa9ae',width=5)
        # Clip the local projection to its own viewport; preserve the recorded coordinates and geometry.
        local=im.crop((776,165,1078,362));ld=ImageDraw.Draw(local);ld.line([(gg(p[0])-776,zz(p[1])-165) for p in f['arm']],fill=green,width=5);xx,yy=gg(s)-776,zz(z)-165;ld.ellipse((xx-5,yy-5,xx+5,yy+5),fill=blue);im.paste(local,(776,165));dr=ImageDraw.Draw(im)
        label(dr,781,374,'Bend: %.1f deg'%f['theta'],16,green); label(dr,780,397,'Blue: vehicle   Green: arm',14); x,y=xt(f['t']),ya(f['theta']);dr.ellipse((x-4,y-4,x+4,y+4),fill=orange)
        if i==round(34.084/speed*rate):im.save(A/'replay-poster.png')
        ff.stdin.write(im.tobytes())
    ff.stdin.close(); assert ff.wait()==0
    for m in ('fixed','hover'):
        orig=root/(m+'.mp4');orig.write_bytes(blob(c,'revision/data/super_backend/'+m+'/recorded_dynamics.mp4'));sp.run(['ffmpeg','-loglevel','error','-y','-i',str(orig),'-an','-c:v','libx264','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(C/(m+'.mp4'))],check=True)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.1)); ax,bx=axes
    for m,col,ls in [('fixed',blue,'-'),('hover',orange,':')]:
        raw=blob(c,'revision/data/super_backend/'+m+'/states.csv').decode();rows=list(csv.DictReader(io.StringIO(raw))); ts=[float(q['time_s']) for q in rows];cur=[float(q['total_curvature_deg']) for q in rows];cm=[float(q['command_theta_deg']) for q in rows];
        if m=='fixed':ax.plot(ts,cm,'--',color='#69737a',lw=1.05,label='Command')
        ax.plot(ts,cur,ls,color=col,lw=1.6,label={'fixed':'Fixed base','hover':'Hover'}[m]);
        if m=='hover':
            p0=[float(rows[0][k]) for k in ('uav_x','uav_y','uav_z')]; e=[1000*math.sqrt(sum((float(q[k])-p)**2 for k,p in zip(('uav_x','uav_y','uav_z'),p0))) for q in rows];bx.plot(ts,e,c=green,lw=1.4);bx.set_ylabel('Displacement from initial position (mm)')
    ax.set_ylabel('Bending angle (deg)');ax.axhline(120,c='#b55b42',ls=(0,(3,3)),lw=.65);ax.legend(loc='lower center',ncol=3,fontsize=7,bbox_to_anchor=(.5,-.41),frameon=False);ax.set_title('(a)  Command and integrated shape',loc='left',fontsize=11);bx.set_title('(b)  Hover displacement',loc='left',fontsize=11)
    for p in axes:p.spines['top'].set_visible(False);p.spines['right'].set_visible(False);p.set_xlabel('Simulation time (s)');p.grid(axis='y',color=rule,linewidth=.5);p.set_xlim(0,14)
    fig.subplots_adjust(left=.08,right=.98,top=.87,bottom=.25,wspace=.39);fig.savefig(C/'response.png',dpi=200);fig.savefig(C/'response.pdf');plt.close(fig)
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
        for part in [A,C]:
            for p in sorted(part.iterdir()):tar.add(p,arcname=part.name+'/'+p.name,recursive=False)
