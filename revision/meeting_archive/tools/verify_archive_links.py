"""Check every archive link offline, including the HTML inside each downloadable ZIP."""
import argparse, hashlib, json, zipfile, posixpath
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote
class Links(HTMLParser):
 def __init__(self):super().__init__();self.refs=[];self.ids=set()
 def handle_starttag(self,t,attrs):
  d=dict(attrs)
  if d.get('id'):self.ids.add(d['id'])
  for k in ('href','src','poster'):
   if d.get(k):self.refs.append(d[k])
def inspect(raw):
 out=Links();out.feed(raw.decode('utf-8'));return out
def check(raw,path,exists):
 tree=inspect(raw)
 for href in tree.refs:
  parsed=urlsplit(href)
  if parsed.scheme or parsed.netloc or parsed.path.startswith('/'):raise ValueError(f'{path}: not self-contained: {href}')
  target=posixpath.normpath(posixpath.join(posixpath.dirname(path),unquote(parsed.path))) if parsed.path else path
  if not exists(target):raise ValueError(f'{path}: broken: {href}')
  if not parsed.path and parsed.fragment and unquote(parsed.fragment) not in tree.ids:raise ValueError(f'{path}: missing section: {href}')
 return len(tree.refs)
def verify(site):
 n=0;media_count=0;folder=site/'archive'; pages=sorted(folder.rglob('index.html'))
 assert len(pages)>1
 for html in pages:n+=check(html.read_bytes(),html.relative_to(site).as_posix(),lambda name:(site/name).is_file())
 for zpath in folder.glob('*/meeting.zip'):
  with zipfile.ZipFile(zpath) as z:
   names=set(z.namelist());n+=check(z.read('index.html'),'index.html',lambda name:name in names);manifest=json.loads(z.read('files.json'))
   source=json.loads((zpath.parent/'snapshot.json').read_bytes());expected={f['file'] for f in source['files']};found={f['file'] for f in manifest['files']};assert found|set(manifest['separate_files'])==expected and not (found&set(manifest['separate_files']))
   for f in manifest['files']:
    assert hashlib.sha256(z.read('media/'+f['file'])).hexdigest()==f['sha256'];media_count+=1
 print(f'Links verified: {n}; web pages: {len(pages)}; offline package assets: {media_count}')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--site',type=Path,required=True);verify(ap.parse_args().site.resolve())
