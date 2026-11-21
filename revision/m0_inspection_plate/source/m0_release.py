"""Mirror exactly the M0 figures, audit receipts, code and offline presentation."""
from pathlib import Path
import tarfile
import zipfile
from m0_data import ROOT,OUT,QA,WORK,read_json,write_json,sha_file
from m0_archive import ID,CAN,STEMS

def main():
    source=ROOT/'source'
    archive=CAN/'dist/archive'/ID
    snap=read_json(archive/'snapshot.json')
    for f in snap['files']:
        p=archive/'media'/f['file'];assert f['sha256']==sha_file(p) and f['bytes']==p.stat().st_size
    for f in snap['outputs'].values():assert f['sha256']==sha_file(archive/f['file'])
    with zipfile.ZipFile(archive/'meeting.zip') as z:
        assert z.testzip() is None and {'index.html','media/replay.mp4','media/review.pdf'}.issubset(z.namelist())
    rep=[p for s in STEMS for p in [OUT/(s+ext) for ext in ('.png','.svg','.pdf')]]
    rep+=[OUT/'gate_review_v2.pdf',OUT/'gate_multiview_v2.mp4']
    qa=[QA/(n+'.json') for n in ('archive-fingerprints','scene-clearance','hero-projection','inset-projection',
                                  'pathtrace','vector-verification','video-verification','review-verification')]
    qa += [QA/'machined-portal-proof.png']
    stage=[ROOT/'README.md',*sorted(source.glob('*.py')),*rep,*qa]
    paths={'revision/m0_inspection_plate/'+p.relative_to(ROOT).as_posix():p for p in stage}
    paths.update({'revision/meeting_archive/archive/'+ID+'/'+p.relative_to(archive).as_posix():p
                  for p in archive.rglob('*') if p.is_file()})
    receipt=QA/'release-manifest.json'
    write_json(receipt,{'source_commit':'951c512fbeaf860c5158e18c44aed527dbf7a899','files':{
        n:{'sha256':sha_file(p),'bytes':p.stat().st_size} for n,p in paths.items()}})
    paths['revision/m0_inspection_plate/qa/release-manifest.json']=receipt
    with tarfile.open(WORK/'nuc-release.tgz','w:gz',compresslevel=1) as bundle:
        for name,p in sorted(paths.items()):bundle.add(p,arcname=name,recursive=False)
    print(len(paths),'files for NUC; transfer', (WORK/'nuc-release.tgz').stat().st_size,flush=True)
if __name__=='__main__':main()
