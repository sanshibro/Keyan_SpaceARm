"""World-fixed synchronized CAD replay. Only propeller phase is illustrative."""
import argparse
from pathlib import Path
import json
import subprocess
import tempfile

import numpy as np
import pandas as pd
from PIL import Image,ImageDraw

from round_f_plant import Plant,LAUNCH
from round_f_rotor_replay import View,split_model,font,TURNS_PER_SIM_SECOND
from round_fp_run import P

SEED=P['development_seed']
FPS=16
COLORS={'straight':'#66737a','nominal':'#345f84','fastbend':'#a35432','slow':'#26746c'}
INK='#233c48'
GRAY='#51646e'
LINE='#d4dde0'
W,H=1328,650
CELL=416
X=(23,456,889)


def project(view,world):
    left,right=view.render.scene.camera
    eye=(np.asarray(left.pos)+right.pos)/2
    dx=np.asarray(world)-eye
    size=left.frustum_top-left.frustum_bottom
    return np.array([240+480*(dx@np.cross(left.forward,left.up))/size,240-480*(dx@left.up)/size])


class Flight:
    def __init__(self,root,key):
        folder=root/'runs'/('%d-%s'%(SEED,key))
        self.raw=np.load(str(folder/'physics.npz'))
        self.release=json.loads((folder/'run.json').read_text())['release_us']
        w=pd.read_csv(folder/'hil.csv')
        self.ts=w.us.to_numpy()
        self.angles=w.net_deg.to_numpy()
        self.pos=w[['ex','ey','ez']].values+LAUNCH
        self.ref=LAUNCH+np.array(P['target_ENU_m'])
        self.error=100*np.linalg.norm(self.pos-self.ref,axis=1)
        drive=w[['m1','m2','m3','m4']].values
        self.phase=np.vstack([np.zeros(4),np.cumsum((np.sqrt(drive[:-1])*[1,1,-1,-1])*
                    (np.diff(self.ts)*2*np.pi*TURNS_PER_SIM_SECOND/1e6)[:,None],axis=0)])

    def at(self,t,view):
        u=self.release+int(round(t*1e6))
        j=int(np.abs(self.raw['us']-u).argmin())
        k=int(np.abs(self.ts-u).argmin())
        img,_=view.draw(self.raw['qpos'][j],self.phase[k],False,anchor=self.ref,side_span=.95)
        tip_to_ref=project(view,self.ref)
        dims=img.size[0]
        draw=ImageDraw.Draw(img)
        x,y=tip_to_ref
        for a,b,c,d in [(x-12,y,x-4,y),(x+4,y,x+12,y),(x,y-12,x,y-4),(x,y+4,x,y+12)]:
            draw.line((a,b,c,d),fill='#964b30',width=2)
        # Draw the last 1.5 seconds of actual body motion in fixed world coordinates.
        trail=[tuple(project(view,self.pos[i])) for i in range(max(0,k-375),k+1,10)]
        if len(trail)>1:draw.line(trail,fill='#27817f',width=2)
        a,b=project(view,self.raw['qpos'][j,:3]);draw.ellipse((a-3,b-3,a+3,b+3),fill='#175c65')
        return img,self.angles[k],self.error[k],abs(int(self.raw['us'][j])-u)


def main(root):
    out=root/'review';out.mkdir(exist_ok=True)
    frames=0;max_skew=0
    flights={key:Flight(root,key) for key in COLORS}
    ft=[font(27,True),font(20,True),font(18),font(16),font(25)]
    with tempfile.TemporaryDirectory(prefix='fp-rotors-') as temp:
        plant=Plant()
        model,axes,_=split_model(plant,Path(temp));view=View(model,axes)
        cmd=['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','%dx%d'%(W,H),
             '-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',
             '-movflags','+faststart',str(out/'comparison.mp4')]
        ff=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        scenes=[('主动弯臂','实际执行同一条弯曲路径，比较 6 秒与 3 秒到达目标弯角。',
                 ('straight','nominal','fastbend'),('直臂定点','常速弯臂 · 6 s','快速弯臂 · 3 s'),15.,24.,16.,20.,'bend'),
                ('收臂恢复','相同弯臂与保持过程后，比较 4 秒与 8 秒收回。',
                 ('straight','nominal','slow'),('直臂定点','常速收臂 · 4 s','慢速收臂 · 8 s'),23.,35.,24.,28.,'return')]
        try:
            for title,note,keys,labels,begin,end,zero,still,key in scenes:
                for i in range(round((end-begin)*FPS)+1):
                    t=begin+i/FPS
                    canvas=Image.new('RGB',(W,H),'white');draw=ImageDraw.Draw(canvas)
                    draw.text((23,12),title,font=ft[0],fill=INK);draw.text((195,21),note,font=ft[2],fill=GRAY)
                    draw.text((1087,21),'PX4 SITL  /  MuJoCo',font=ft[2],fill=GRAY)
                    draw.line((23,62,1304,62),fill=LINE,width=2)
                    for j,(x,v,name) in enumerate(zip(X,keys,labels)):
                        img,angle,err,mis=flights[v].at(t,view);max_skew=max(max_skew,mis)
                        draw.line((x,86,x+22,86),fill=COLORS[v],width=5)
                        draw.text((x+31,72),name,fill=COLORS[v],font=ft[1])
                        canvas.paste(img.resize((CELL,CELL),Image.LANCZOS),(x,105))
                        draw.text((x,525),'实测仿真弯角  %.1f°'%angle,font=ft[2],fill=GRAY)
                        draw.text((x+185,520),'定点偏差  %.1f cm'%err,font=ft[4],fill=COLORS[v])
                    draw.line((23,572,1304,572),fill=LINE,width=2)
                    draw.text((23,581),'动作起点  %+.2f s'%(t-zero),font=ft[1],fill=INK)
                    draw.text((288,586),'共同固定世界机位  ·  十字为定点  ·  圆点为机身，细线为最近 1.5 秒轨迹',font=ft[3],fill=GRAY)
                    draw.text((1040,586),'配对种子  %d'%SEED,font=ft[3],fill=GRAY)
                    draw.text((23,619),'姿态来自积分状态；桨叶为四路推力驱动的降速显示。两段按仿真时间 1× 播放。',font=ft[3],fill=GRAY)
                    if abs(t-still)<1e-6:canvas.save(out/(key+'.png'))
                    ff.stdin.write(canvas.tobytes());frames+=1
        finally:
            ff.stdin.close();status=ff.wait();view.render.close()
        if status:raise RuntimeError('ffmpeg failed')
    (out/'video.json').write_text(json.dumps(dict(seed=SEED,fps=FPS,frames=frames,playback=1,
        max_qpos_time_error_ms=max_skew/1000,world_fixed_anchor=(LAUNCH+P['target_ENU_m']).tolist(),
        native_meshes=True,propeller_phase='Integrated signed square root of per-channel thrust; illustrative speed, no measured RPM.'),indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('round',type=Path);main(ap.parse_args().round)
