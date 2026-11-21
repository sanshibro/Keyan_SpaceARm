"""Save three original wrist exposures and a release-clock HIL trace from a development flight."""
import argparse,csv,hashlib,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import numpy as np

def snapshot(f,out):
    out.mkdir(parents=True,exist_ok=True)
    optics=f/'optics'
    stream=[json.loads(line) for line in (optics/'index.jsonl').read_text().splitlines() if line.endswith('}')]
    events=[json.loads(line) for line in (f/'events.jsonl').read_text().splitlines() if line.endswith('}')]
    release=next(e['us'] for e in events if e['event']=='support_removed')
    ts=[entry['t'] for entry in stream]
    if len(ts)<3:raise ValueError('insufficient exposures')
    times=[max(ts[0],min(20.,ts[-1]*.5)),(max(20.,ts[0])+ts[-1])*.5,ts[-1]]
    chosen=[stream[int(np.argmin(np.abs(np.asarray(ts)-t)))] for t in times]
    with (f/'hil.csv').open() as file:
        reader=csv.DictReader(file);hil=[]
        for row in reader:
            if row['released']!='1':continue
            try:hil.append([1.e-6*(int(row['us'])-release),float(row['issued_deg']),float(row['net_deg'])])
            except (ValueError,TypeError,KeyError):continue
    sheet=Image.new('RGB',(1440,430),'white');painter=ImageDraw.Draw(sheet)
    try:font=ImageFont.truetype('DejaVuSans.ttf',17)
    except OSError:font=ImageFont.load_default()
    for i,row in enumerate(chosen):
        file=optics/row['rgb']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=row['rgb_sha256']:raise ValueError('wrist hash')
        im=Image.open(file).convert('RGB');sheet.paste(im,(i*480,0))
        painter.text((i*480+12,370),'Wrist RGB   t = %.2f s    frame %d'%(row['t'],row['frame']),fill='#263c42',font=font)
    sheet.save(out/'pilot-wrist.jpg',quality=94)
    (out/'snapshot.json').write_text(json.dumps(dict(progress=json.loads((f/'progress.json').read_text()),release_us=release,frames=chosen,last_exposure_s=ts[-1],hil_samples=len(hil)),indent=2)+'\n')
    if hil:np.savez_compressed(out/'hil-trace.npz',t=np.array(hil)[:,0],issued_deg=np.array(hil)[:,1],net_deg=np.array(hil)[:,2])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('flight',type=Path);p.add_argument('output',type=Path);a=p.parse_args();snapshot(a.flight,a.output)
