"""Record local source and dependency identity for development runs."""
from pathlib import Path
import hashlib,json,shutil,subprocess,platform,os,importlib

def freeze_run(run):
    root=Path(__file__).resolve().parent;run=Path(run);dst=run/'source_snapshot';dst.mkdir(exist_ok=True)
    hashes={}
    for p in sorted(root.glob('*.py')):
        shutil.copyfile(p,dst/p.name);hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    v2_root=root.parent/'xtdrone_catkin_ws/src/uav_continuum_whole_body_v2'
    v2_dst=dst/'v2';v2_dst.mkdir(exist_ok=True);v2_hashes={}
    for p in sorted((v2_root/'src/uav_continuum_whole_body_v2').glob('*.py')):
        shutil.copyfile(p,v2_dst/p.name);v2_hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    config=v2_root/'config/planner_v2_realtime.yaml'
    shutil.copyfile(config,v2_dst/config.name)
    v2_hashes[config.name]=hashlib.sha256(config.read_bytes()).hexdigest()
    sdk_dst=dst/'qingyu_config';sdk_dst.mkdir(exist_ok=True)
    for relative in ('PROVENANCE.json','config/continuum_arm_visualize.yaml'):
        source=root/'vendor/qingyu_continuum_arm'/relative
        shutil.copyfile(source,sdk_dst/source.name)
    dependencies={}
    for p in sorted((root.parent/'.mujoco_deps').glob('*.dist-info/METADATA')):
        fields={}
        for line in p.read_text().splitlines():
            if line.startswith(('Name: ','Version: ','License: ')):
                k,v=line.split(': ',1);fields[k]=v
        dependencies[p.parent.name]=fields
    manifest={'source_files_sha256':hashes,'reused_v2_files_sha256':v2_hashes,'installed_dependencies':dependencies,'base_git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(root),text=True).strip(),'scope':'local source snapshot; vendor CAD provenance in vendor/qingyu_continuum_arm'}
    versions={}
    for name in ('mujoco','numpy','scipy','cv2','yaml'):
        try:versions[name]=getattr(importlib.import_module(name),'__version__','unknown')
        except ImportError:versions[name]='unavailable'
    manifest['runtime']=dict(python=platform.python_version(),platform=platform.platform(),logical_cpu_count=os.cpu_count(),versions=versions,thread_settings={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
    (run/'source_dependencies.json').write_text(json.dumps(manifest,indent=2))
