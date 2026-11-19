"""Make space for the new video and paper in landing-page PNGs; verify RGBA pixels and metadata."""
import io,sys
from pathlib import Path
from PIL import Image
def keep(p,raw):
    with Image.open(p) as old,Image.open(io.BytesIO(raw)) as new:
        if old.convert('RGBA').tobytes()!=new.convert('RGBA').tobytes() or old.size!=new.size:return False
        return all(new.info.get(k)==v for k,v in old.info.items() if k!='transparency')
def run(stage):
    sys.path.insert(0,str(stage/'qa/packing'))
    from zopfli.png import optimize
    root=stage/'site-cache/dist'
    files=list(root.rglob('*'));old=sum(p.stat().st_size for p in files if p.is_file())
    target=256*1024**2-65536 if (root/'archive/2026-11-19-confidence').is_dir() else 256*1024**2-610000
    required=max(0,old-target);saved=0
    names=['super_body_camera.png','super_hover_frame.png','super_platform_preview.png','super_distal_tool.png','cad16_scene_preview.png']
    paths=[root/n for n in names];paths.extend(sorted((p for p in root.glob('*.png') if p not in paths),key=lambda p:-p.stat().st_size))
    for p in paths:
        if saved>=required:break
        initial=p.read_bytes()
        raw=optimize(initial,keepchunks=['pHYs','iCCP','tEXt','zTXt','iTXt','gAMA','sRGB','cHRM'],
             num_iterations=4,num_iterations_large=3,filter_strategies='01234m')
        if len(raw)>=len(initial) or not keep(p,raw):continue
        p.write_bytes(raw);saved+=len(initial)-len(raw)
        print(p.name,'saved',len(initial)-len(raw),'cumulative',saved,flush=True)
    print('Landing assets only. Prior archive snapshots are byte-preserved. Freed',saved,'target',required,flush=True)
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
