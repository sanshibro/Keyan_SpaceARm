"""Losslessly compact only older top-level illustrations, leaving archival copies intact."""
import argparse, io, json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
def optimize_file(path,library):
    from zopfli.png import optimize
    original=path.read_bytes()
    candidate=optimize(original,keepchunks=['pHYs','iCCP','tEXt','zTXt','iTXt','gAMA','sRGB','cHRM'],
                       num_iterations=2,num_iterations_large=2,filter_strategies='m')
    if len(candidate)>=len(original):return 0
    with Image.open(io.BytesIO(original)) as a,Image.open(io.BytesIO(candidate)) as b:
        if a.size!=b.size or a.convert('RGBA').tobytes()!=b.convert('RGBA').tobytes():return 0
        if any(b.info.get(k)!=v for k,v in a.info.items() if k!='transparency'):return 0
    path.write_bytes(candidate)
    return len(original)-len(candidate)
def run(stage,library):
    root=stage/'site-cache/dist';sys.path.insert(0,str(library))
    names=['round_d_gate','round_fp_cad','round_d_sweep','passage_mechanism_v2','round_f_curves','round_f_cad','super_backend_checks','round_e1_curves','round_e1_cad','round_e2_cad','bending_envelope_check','round_e2_curves','round_e_curves']
    paths=[root/(n+'.png') for n in names if (root/(n+'.png')).exists()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={p:pool.submit(optimize_file,p,library) for p in paths}
        saved={p.name:f.result() for p,f in jobs.items()}
    print(saved,'Total recovered',sum(saved.values()),flush=True)
if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
