"""Serve large G1 logs from the same pinned public research blobs, preserving exact offline packs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def read(p):return json.loads(p.read_bytes())


def write(p,d):p.write_bytes((json.dumps(d,ensure_ascii=False,indent=2)+'\n').encode())


def main(stage,site):
    tag='2026-11-19-g1';folder=site/'dist/archive'/tag
    native=stage/'site/.sites-checkout/dist/archive'/tag
    links=read(stage/'public-log-links.json');assert len(links)==13
    public_mirror='https://raw.githubusercontent.com/sanshibro/Keyan_SpaceARm/f7df8c87e4439427066e2abfba239c3bb3d96e57/revision/meeting_archive/archive/'+tag+'/'
    links['all-traces.npz']=public_mirror+'media/all-traces.npz'
    canonical=read(native/'snapshot.json')
    text=(native/'index.html').read_text(encoding='utf8')
    cache=site/'source-evidence'/tag;cache.mkdir(parents=True,exist_ok=True)
    specpath=site/'tools/rounds/g1.json';spec=read(specpath)
    for f in canonical['files']:
        name=f['file']
        if name not in links:continue
        url=links[name]
        if name.endswith('.tar.gz'):
            assert url=='https://raw.githubusercontent.com/sanshibro/Keyan_SpaceARm/2976b304453d25d74b01f2c02709e2ecec90b5fb/revision/data/round_g1/evidence/'+name
        original=folder/'media'/name
        if original.exists():
            assert hashlib.sha256(original.read_bytes()).hexdigest()==f['sha256']
            shutil.move(str(original),str(cache/name))
        assert hashlib.sha256((cache/name).read_bytes()).hexdigest()==f['sha256']
        assert 'href="media/'+name+'"' in text
        text=text.replace('href="media/'+name+'"','href="'+url+'"')
        f['external_url']=url
        next(entry for entry in spec['files'] if entry['file']==name)['external_url']=url
    pack='meeting.zip'
    if (folder/pack).exists():
        assert (folder/pack).read_bytes()==(native/pack).read_bytes()
        shutil.move(str(folder/pack),str(cache/pack))
    assert (cache/pack).read_bytes()==(native/pack).read_bytes()
    assert 'href="meeting.zip"' in text
    text=text.replace('href="meeting.zip"','href="'+public_mirror+pack+'"')
    canonical['outputs']['pack']['external_url']=public_mirror+pack
    raw=text.encode();(folder/'index.html').write_bytes(raw)
    canonical['outputs']['page']['sha256']=hashlib.sha256(raw).hexdigest()
    write(folder/'snapshot.json',canonical);write(specpath,spec)
    gen=site/'tools/meeting_archive.py'
    g=gen.read_text(encoding='utf8')
    old="b=(folder/'media'/f['file']).read_bytes();req("
    new="""p=folder/'media'/f['file']
        if f.get('external_url') and not p.is_file():
            p=folder.parents[2]/'source-evidence'/folder.name/f['file']
        b=p.read_bytes();req("""
    if old in g:g=g.replace(old,new,1)
    else:assert new in g
    old_set="expected={f['file'] for f in s['files']};req(expected=="
    new_set="expected={f['file'] for f in s['files'] if not f.get('external_url')};req(expected=="
    if old_set in g:g=g.replace(old_set,new_set,1)
    else:assert new_set in g
    a="path='media/'+f['file']; title_e="
    b="path=e(f['external_url']) if (not offline and f.get('external_url')) else 'media/'+f['file']; title_e="
    if a in g:g=g.replace(a,b,1)
    else:assert b in g
    a="entry=s['outputs'][k];req(sha((folder/entry['file']).read_bytes())=="
    b="""entry=s['outputs'][k];target=folder/entry['file']
        if entry.get('external_url') and not target.is_file():
            target=folder.parents[2]/'source-evidence'/folder.name/entry['file']
        req(sha(target.read_bytes())=="""
    if a in g:g=g.replace(a,b,1)
    else:assert b in g
    a="with zipfile.ZipFile(folder/'meeting.zip') as z:"
    b="with zipfile.ZipFile(target) as z:"
    if a in g:g=g.replace(a,b,1)
    else:assert b in g
    gen.write_bytes(g.encode())
    # The page itself has complete HTTPS links. The redirect map also documents
    # the mapping for static hosts that support location redirects.
    redir=site/'dist/_redirects'
    old=redir.read_text(encoding='utf8').splitlines() if redir.exists() else []
    for n,u in links.items():
        rule='/archive/'+tag+'/media/'+n+' '+u+' 302'
        if rule not in old:old.append(rule)
    rule='/archive/'+tag+'/meeting.zip '+public_mirror+'meeting.zip 302'
    if rule not in old:old.append(rule)
    redir.write_bytes(('\n'.join(old)+'\n').encode())
    subprocess.run([sys.executable,str(gen),'verify','--site',str(site/'dist')],check=True)
    assert (cache/'meeting.zip').read_bytes()==(native/'meeting.zip').read_bytes()
    for older in (native.parent).glob('*/snapshot.json'):
        if older.parent.name==tag:continue
        assert older.read_bytes()==(folder.parent/older.parent.name/'snapshot.json').read_bytes()
    print('All thirteen public log links verified; ten preceding manifests and the offline packet are byte-identical.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',type=Path);ap.add_argument('site',type=Path)
    a=ap.parse_args();main(a.stage,a.site)
