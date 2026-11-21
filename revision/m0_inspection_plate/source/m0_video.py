"""1920x1080 synchronized review of five original flights, including the delivered RGB."""
import json,math,subprocess
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from matplotlib.font_manager import findfont
from m0_data import OUT,QA,selected,plant,LABELS,COLORS,write_json,sha_file
from m0_render import Renderer,SIDE,GATE_CAM,DOCK_CAM
from m0_scene import A,B,C

W,H=1920,1080
START,STOP,FPS=23.,55.,15
METHODS=("vision","strain","fixed","proposed","oracle")
COL=361;LEFT=36;GAP=11
WP=657
INK="#263e42";MUTED="#596e73";RED="#a0523f";TEAL="#147968";TAN="#ab713f"
def font(sz,bold=False,serif=False):
    name=('Times New Roman' if serif else 'Arial')
    from matplotlib.font_manager import FontProperties
    return ImageFont.truetype(findfont(FontProperties(family=name,weight='bold' if bold else 'normal')),sz)
F14,F16,F18,F20,F22,F30=font(15),font(17),font(19),font(21,True),font(23,True),font(31,True,True)
def xcol(j):return LEFT+j*(COL+GAP)
def fit_frame(stream,row):
    im=stream.image(row);im.thumbnail((COL-8,157),Image.Resampling.LANCZOS);return im
def world_line(canvas,cam,size,points,col,lw,dash=False):
    d=ImageDraw.Draw(canvas);uv=cam.pixels(points,size)
    for a,b in zip(uv,uv[1:]):
        if dash:
            l=np.linalg.norm(b-a);n=max(1,int(l/7))
            for j in range(0,n,2):
                p=a+(b-a)*j/n;q=a+(b-a)*min(n,j+1)/n
                d.line([tuple(p),tuple(q)],fill=col,width=lw)
        else:d.line([tuple(a),tuple(b)],fill=col,width=lw)
    return uv
def world_frame(r,f,now,cam,size,detail=False):
    frame=r.render(cam,size);d=ImageDraw.Draw(frame);i=f.row(now);z=f.z
    if cam is DOCK_CAM:
        tip=z['tip'][i];u=z['tool_axis'][i];th=np.linspace(0,2*np.pi,91)
        ring=C+.035*(B[:,0]*np.cos(th)[:,None]+B[:,1]*np.sin(th)[:,None])
        world_line(frame,cam,size,ring,TAN,2)
        world_line(frame,cam,size,[C-.15*A,C+.025*A],TAN,3,True)
        uv=world_line(frame,cam,size,[tip-.145*u,tip],TEAL,3);p=uv[-1]
        d.ellipse((p[0]-3,p[1]-3,p[0]+3,p[1]+3),fill=TEAL,outline='white')
    elif detail and z['arm_tool_clearance'][i]<.5:
        pts=np.stack([z['arm_near_robot'][i],z['arm_near_gate'][i]])
        uv=world_line(frame,cam,size,pts,RED,3)
        for u,v in uv:d.ellipse((u-3,v-3,u+3,v+3),fill='white',outline=RED,width=2)
    return frame
def bar(d,x,y,name,val,lim,cap,unit,low=False):
    d.text((x,y),name,font=F14,fill=MUTED);d.text((x+COL-12,y),f"{val:.1f} {unit}",anchor="ra",font=F16,fill=INK)
    a=x+1;b=x+COL-13;yy=y+22;d.line((a,yy,b,yy),fill="#d9e1df",width=3)
    bad=val<lim if low else val>lim
    d.line((a,yy,a+(b-a)*min(1,max(val/cap,0)),yy),fill=RED if bad else TEAL,width=3)
    t=a+(b-a)*lim/cap;d.line((t,yy-4,t,yy+4),fill="#806750",width=1)
def frame(now,r,flights,streams):
    im=Image.new("RGB",(W,H),"white");d=ImageDraw.Draw(im)
    d.text((LEFT,16),"Inspection through a bulkhead aperture",font=F30,fill=INK)
    d.text((LEFT,57),"Differential strain offset  /  five independent flights  /  seed 26112131",font=F18,fill=MUTED)
    d.text((W-39,19),f"{now:05.2f} s",anchor="ra",font=F30,fill=INK)
    d.text((W-39,59),"time after release",anchor="ra",font=F16,fill=MUTED)
    d.line((LEFT,90,W-36,90),fill="#a8b8b6",width=1)
    for j,m in enumerate(METHODS):
        f=flights[m];z=f.z;i=f.row(now);x=xcol(j);r.set_pose(f.pose(now))
        d.text((x,102),LABELS[m],font=F22,fill=("#77909b" if m=="vision" else COLORS[m]))
        r1=world_frame(r,f,now,SIDE,(544,326));r1=r1.resize((COL,216),Image.Resampling.LANCZOS);im.paste(r1,(x,139))
        d.text((x,360),"Passage  /  side elevation",font=F14,fill=MUTED)
        cam=DOCK_CAM if now>=39. else GATE_CAM
        r2=world_frame(r,f,now,cam,(544,326),True).resize((COL,216),Image.Resampling.LANCZOS);im.paste(r2,(x,390))
        d.text((x,612),"Port axes  /  oblique" if cam is DOCK_CAM else "Arm–sill witnesses  /  section",font=F14,fill=MUTED)
        native=streams[m].at(now);img=fit_frame(streams[m],native)
        im.paste(img,(x+(COL-img.width)//2,WP))
        d.text((x,818),f"Wrist RGB    {native['t']:.2f} → {native['arrival_t']:.2f} s",font=F14,fill=MUTED)
        hold=str(z['hold_reason'][i])
        label=("HOLD · "+hold.replace('_',' ')) if z['hold'][i] else str(z['phase'][i]).replace('_',' ').upper()
        d.text((x,851),label,font=F20,fill=RED if z['hold'][i] else INK)
        bar(d,x,887,'Arm / frame',100*z['arm_tool_clearance'][i],6.,30.,'cm',True)
        bar(d,x,925,'Airframe / frame',100*z['body_clearance'][i],6.,30.,'cm',True)
        bar(d,x,963,'Tool position',100*z['tip_error'][i],3.5,15.,'cm')
        bar(d,x,1001,'Tool axis',float(z['axis_error_deg'][i]),3.,15.,'°')
    d.line((LEFT,1038,W-36,1038),fill="#c4ceca")
    d.text((LEFT,1045),"Wrist exposure blocked 28–32 s   ·   Frame delivery +0.20 s   ·   Clearance ≥ 6.0 cm   ·   Docking ≤ 3.5 cm and ≤ 3.0° for 1 s",font=F16,fill=MUTED)
    return im

def main():
    flights=selected();streams={m:f.original_wrist() for m,f in flights.items()};r=Renderer(plant())
    file=OUT/'gate_multiview_v2.mp4';partial=OUT/'gate_multiview_v2.part.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}',
         '-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p',
         '-movflags','+faststart',str(partial)]
    start=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for k in range(round((STOP-START)*FPS)+1):
            now=START+k/FPS;im=frame(now,r,flights,streams);start.stdin.write(im.tobytes())
            if k in (0,105,240,435):
                im.save(QA/f'film-{k:03}.jpg',quality=91)
            if k%75==0:print('Frame',k,flush=True)
    except BaseException:start.kill();raise
    finally:start.stdin.close();rc=start.wait()
    if rc:raise RuntimeError('Video encoder: '+str(rc))
    reader=imageio_ffmpeg.read_frames(str(partial),pix_fmt='rgb24');meta=next(reader)
    assert tuple(meta['size'])==(W,H) and abs(meta['fps']-FPS)<.01
    blacks={m:0 for m in METHODS};errors={m:[] for m in METHODS};count=0
    for k,buf in enumerate(reader):
        count+=1;a=np.frombuffer(buf,'u1').reshape(H,W,3);t=START+k/FPS
        for j,m in enumerate(METHODS):
            row=streams[m].at(t);original=fit_frame(streams[m],row);x=xcol(j)+(COL-original.width)//2
            pixels=a[WP+3:WP+original.height-3,x+3:x+original.width-3]
            expected=np.asarray(original)[3:-3,3:-3];err=float(np.abs(pixels.astype('i2')-expected).mean())
            if err>9:raise ValueError(f'{m} archive tile mismatch at {t}: {err}')
            errors[m].append(err)
            if row['mask_fraction']>=1-1e-8:
                assert expected.max()==0 and pixels.mean()<1
                blacks[m]+=1
    assert count==1+round((STOP-START)*FPS) and all(x==4*FPS for x in blacks.values())
    partial.replace(file)
    write_json(QA/'video-verification.json',dict(file=file.name,sha256=sha_file(file),width=W,height=H,
        fps=FPS,frames=count,release_s=[START,STOP],checked_native_wrist_tiles=count*5,black_frames=blacks,
        highest_pixel_mae_8bit={m:max(x) for m,x in errors.items()},source_pose="nearest archived qpos"))
    print('Encoded and verified',count*5,'native views',flush=True)
if __name__=='__main__':main()
