"""Read plateau motion in the manufacturer's embedded repeatability plot, in pixel units."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from pypdf import PdfReader
def main(source,output):
    if hashlib.sha256(source.read_bytes()).hexdigest()!='c9f9dde92546d43e57a11433b03a99e27fd39ef4d79810fd0fb037a0840d80e3':
        raise ValueError('Digitization windows belong to the archived SLIM datasheet only')
    images=[o.image.convert('L') for o in PdfReader(source).pages[1].images if o.image.size==(411,353)]
    assert len(images)==1
    raster=np.asarray(images[0]); rows=[]
    for level,sequence,xlo,xhi in [(50,'before80',190,201),(50,'before80',213,224),(50,'before80',235,246),(80,'peak',257,268),(80,'peak',280,291),(80,'peak',303,314),(50,'after80',325,337),(50,'after80',347,359),(50,'after80',370,382)]:
        ylo,yhi=(60,76) if level==80 else (162,180)
        def centers(xs,bot,top):
            out=[]
            for x in xs:
                y=np.flatnonzero(raster[bot:top,x]<130)
                if not len(y):raise ValueError('Plateau trace not found')
                out.append(float(np.median(y)+bot))
            return out
        first=np.median(centers(range(xlo+2,xlo+5),ylo,yhi))
        last=np.median(centers(range(xhi-4,xhi-1),ylo,yhi))
        baseline=np.median(centers(range(xlo-5,xlo-2),260,279))
        rows.append(dict(level=level,sequence=sequence,x_interior=[xlo+2,xhi-2],early_px=float(first),late_px=float(last),
                         step_height_px=float(baseline-first),relative_sag_pct=float(100*(last-first)/(baseline-first))))
    output.write_bytes((json.dumps({'source':'Elastreme Sense Slim, p. 2, repeatability trace (no numerical R axis or synchronized length input).',
       'trace_reading_uncertainty_px':2,'plateaus':rows},indent=2)+'\n').encode())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('pdf',type=Path);p.add_argument('output',type=Path);args=p.parse_args();main(args.pdf,args.output)
