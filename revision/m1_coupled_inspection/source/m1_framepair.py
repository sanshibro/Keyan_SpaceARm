"""Export a same-exposure two-camera witness from a development or registered flight."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw
from m1_media import Flight,Visual,title
from m1_plan import writej
def main():
    ap=argparse.ArgumentParser();ap.add_argument('case',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--time',type=float,required=True);args=ap.parse_args()
    args.out.mkdir(exist_ok=True,parents=True);v=Visual()
    try:
        f=Flight(args.case,v.p.scene_digest);o,j=f.nearest(args.time)
        observer=v.shot(f.q[j],'observer_dock',small=True)
        wrist=f.wrist(o);page=Image.new('RGB',(1418,565),'white');draw=ImageDraw.Draw(page)
        page.paste(observer,(12,41));page.paste(wrist,(907,105))
        title(draw,(18,5),'External view',23,True,True);title(draw,(910,66),'Native wrist RGB-D',23,True,True)
        title(draw,(18,532),'Same exposure    %.2f s      /      Delivery    %.2f s'%(o['t'],o['arrival_t']),18)
        page.save(args.out/'same-exposure.jpg',quality=93)
        writej(args.out/'exposure.json',{'scene':v.p.scene_digest,'time':o['t'],'delivery':o['arrival_t'],
                 'state_sha256':o['sensor_qpos_sha256'],'rgb_sha256':o['rgb_sha256']})
    finally:v.close()
if __name__=='__main__':main()
