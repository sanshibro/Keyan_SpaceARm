"""Append immutable, self-contained meeting records to a static site (Python >=3.8)."""
import argparse, datetime as dt, hashlib, html, json, os, re, shutil, tempfile, zipfile
from pathlib import Path

SCHEMA=1
GROUPS={'image':('images','模型与场景','原图'),'curve':('curves','曲线与机制图','原图'),'video':('videos','视频回放','MP4'),'document':('documents','论文与报告','PDF'),'data':('data','数据及运行记录','数据')}
EXT={'image':{'.png','.jpg','.jpeg','.webp'},'curve':{'.png','.jpg','.webp'},'video':{'.mp4'},'document':{'.pdf'},'data':{'.csv','.json','.npz','.txt','.ulg','.bag'},'poster':{'.png','.jpg','.webp'}}
SAFE=re.compile(r'^[a-z0-9][a-z0-9-]{0,79}$'); FILE=re.compile(r'^[a-z0-9][a-z0-9._-]{0,95}$')
LIMIT=24*1024**2
CSS='''
:root{--ink:#142e3c;--body:#31464e;--muted:#4b626c;--line:#cbd7d9;--accent:#1e655f;--paper:#fff;--wash:#f3f7f6;color-scheme:light}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:1rem}body{margin:0 auto;max-width:1160px;padding:0 clamp(16px,4vw,52px) 40px;color:var(--body);background:var(--paper);font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}a{color:#175c73;text-underline-offset:.22em;text-decoration-thickness:1px}a:hover{color:#0d3e4c}a:focus-visible{outline:3px solid #b7792c;outline-offset:3px}h1,h2,h3,p,figure,ol{margin:0}h1,h2,h3{color:var(--ink);line-height:1.3}h1{font-family:Georgia,"Noto Serif CJK SC","Songti SC",SimSun,serif;font-size:clamp(31px,4vw,49px);font-weight:650;letter-spacing:-.025em}h2{font-size:clamp(20px,3vw,26px)}h3{font-size:18px}.navtop,footer{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);gap:12px;padding:17px 0;font-size:14px;color:var(--muted)}.eyebrow,.badge{font-size:13px;font-weight:650;color:var(--accent);letter-spacing:.04em}.eyebrow{margin:23px 0 7px}.hero p.desc{margin:12px 0 8px;max-width:51em;font-size:17px}.meta{color:var(--muted);font-size:14px}.hero{padding-bottom:22px}.block{padding-top:24px;margin-top:8px;border-top:1px solid var(--line)}.group-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin:0 0 15px}.btn{display:inline-flex;align-items:center;text-decoration:none;border:1px solid #b2c8c8;border-radius:3px;padding:6px 12px;background:#fff;font-size:14px;font-weight:600;white-space:normal}.btn.primary{background:#1a5656;color:#fff;border-color:#1a5656}.btn:hover{border-color:var(--accent)}.actions,.pills{display:flex;flex-wrap:wrap;gap:8px 12px;margin-top:13px}.pills a{font-size:14px}.read{border-left:3px solid var(--accent);padding:12px 18px;background:var(--wash);margin:5px 0 18px}.read h2{font-size:17px;margin-bottom:5px}.read ol{padding-left:1.45em}.scope{font-size:14px;color:var(--muted);margin-top:8px}.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:17px 22px}.gallery.solo{grid-template-columns:minmax(0,1fr)}.gallery .media{display:flex;align-items:center;justify-content:center;background:#fafbfa;border:1px solid var(--line);padding:6px;min-height:110px}.gallery img,video{display:block;max-width:100%;height:auto}.gallery .media img{width:100%;height:clamp(185px,25vw,305px);object-fit:contain}.gallery.solo .media img{height:clamp(205px,39vw,430px)}figcaption{margin-top:8px}figcaption p,.row p{margin:4px 0;color:var(--muted);font-size:14px;line-height:1.6}.small{font-size:14px}.video{width:100%;background:#eef1f0;max-height:420px;border:1px solid var(--line)}.vidgrid{align-items:start}.role{color:var(--accent);font-size:12px;font-weight:650;margin-right:6px}.list{border-top:1px solid var(--line)}.row{display:flex;align-items:center;justify-content:space-between;gap:15px;padding:10px 2px;border-bottom:1px solid var(--line)}.row strong{color:var(--ink)}.fileact{white-space:nowrap;font-size:14px}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:23px}.card{border-top:2px solid #849c9c;padding:12px 0 14px;display:grid;grid-template-columns:1fr 132px;gap:8px 14px;align-items:start}.card .copy{min-width:0}.card .thumb{width:100%;height:108px;background:#fafbfa;object-fit:contain;border:1px solid var(--line)}.card h3{margin:5px 0;font-size:20px}.card h3 a{color:var(--ink);text-decoration:none}.card h3 a:hover{text-decoration:underline}.card p{font-size:14px;color:var(--muted)}.card .counts{font-size:13px}.card .enter{font-size:15px;font-weight:650}.indexLead{max-width:46em;color:var(--muted);margin:12px 0}.how{font-size:14px;color:var(--muted);margin-top:10px}footer{margin-top:34px;border-top:1px solid var(--line);border-bottom:0;align-items:flex-start}.quick{margin:9px 0 0}.quick a{margin:0 12px 0 0}code{font-size:.9em;overflow-wrap:anywhere}.skip{position:absolute;left:-9999px}.skip:focus{left:12px;top:12px;background:white}.note{font-size:14px;color:var(--muted);margin:5px 0}.navtop strong{font-family:Georgia,SimSun,serif;color:var(--ink);font-size:17px} @media(max-width:760px){.cards{grid-template-columns:1fr}.card{grid-template-columns:1fr 112px}.gallery{grid-template-columns:1fr}.gallery .media img{height:auto;max-height:400px}.row{align-items:flex-start}.navtop,footer{flex-wrap:wrap}} @media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}} @media print{body{max-width:none;padding:0}.navtop,.actions,.pills,.fileact,.quick,footer,.skip{display:none}.gallery{grid-template-columns:1fr 1fr}figure,.row{break-inside:avoid}.read{background:white}}
'''
def e(s):return html.escape(str(s),quote=True)
def jbytes(obj):return (json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode()
def read(path):return json.loads(path.read_bytes())
def sha(b):return hashlib.sha256(b).hexdigest()
def same_json(a,b):return jbytes(a)==jbytes(b)
def req(ok,msg):
    if not ok:raise ValueError(msg)
def size(n):return (f'{n/1024**2:.1f} MB' if n>=1024**2 else f'{max(1,round(n/1024))} KB')
def shell(title,body):return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+e(title)+' · 组会实验记录</title><style>'+CSS+'</style></head><body id="top"><a class="skip" href="#main">跳到内容</a>'+body+'</body></html>\n'
def nav(href,text):return '<nav class="navtop" aria-label="主导航"><strong>空中连续体操作研究</strong><a href="'+href+'">'+text+'</a></nav>'
def counts(s):
    f=s['files'];return ' · '.join(str(sum(x['kind'] in kinds for x in f))+lab for kinds,lab in [(('image','curve'),' 张图'),(('video',),' 段视频'),(('document',),' 份 PDF')] if any(x['kind'] in kinds for x in f))
def anchor(file):return 'file-'+file.replace('.','-')
def detail(s,offline=False,omitted=()):
    ff=[x for x in s['files'] if x['file'] not in omitted]; lookup={f['file']:f for f in ff};d=s['when'];top=nav('#top','本轮离线资料') if offline else nav('../index.html','← 全部轮次'); out=[top,'<main id="main"><header class="hero"><p class="eyebrow">'+e(s['stage'])+'　/　'+e(d[:10])+' '+e(d[11:16])+'（北京时间）</p><h1>'+e(s['title'])+'</h1><p class="desc">'+e(s['summary'])+'</p><p class="meta">'+e(counts({'files':ff}))+'</p>'];
    if not offline:out.append('<div class="actions"><a class="btn primary" href="meeting.zip" download>下载本轮离线汇报包</a><a class="btn" href="snapshot.json" download>下载文件校验清单</a></div>');
    if not offline and s.get('parent_round'):
        parent=s['parent_round'];req(bool(SAFE.fullmatch(parent)) and parent!=s['id'],'invalid parent round')
        out.append('<p><a href="../'+e(parent)+'/index.html">打开对应飞行的原始日志与验收记录</a></p>')
    if omitted:out.append('<p class="note">为方便携带，此包未收入：'+e('、'.join(f['title'] for f in s['files'] if f['file'] in omitted))+'。可在本轮网页单独下载。</p>');
    elif not offline and any(f.get('separate',False) for f in ff):out.append('<p class="note">大附件单独提供：'+e('、'.join(f['title'] for f in ff if f.get('separate',False)))+'。</p>');
    out+=['</header><aside class="read"><h2>组会怎么讲</h2><ol>',''.join('<li>'+e(x)+'</li>' for x in s['guide']),'</ol><p class="scope"><strong>这轮验证了什么：</strong>'+e(s['scope'])+'</p></aside>','<nav class="pills" aria-label="本页内容">']
    for k,(sid,title,_) in GROUPS.items():
        if any(f['kind']==k for f in ff) or k=='video' and s.get('video_note'):out.append('<a href="#'+sid+'">'+title+'</a>')
    out.append('</nav>');
    for k,(sid,title,_) in GROUPS.items():
        entries=[f for f in ff if f['kind']==k]
        if not entries and not (k=='video' and s.get('video_note')):continue
        out.append('<section class="block" id="'+sid+'"><div class="group-head"><h2>'+title+'</h2></div>')
        if k=='video' and s.get('video_note'):out.append('<p class="note">'+e(s['video_note'])+'</p>')
        if entries and k in ('image','curve','video'):out.append('<div class="gallery '+('solo' if len(entries)==1 else '')+' '+('vidgrid' if k=='video' else '')+'">')
        elif entries:out.append('<div class="list">')
        for n,f in enumerate(entries,1):
            path='media/'+f['file']; title_e=e(f['title']);cap=e(f['caption']); sz=size(f['bytes']); at=anchor(f['file'])
            if k in ('image','curve','video'):
                out.append('<figure id="'+at+'">');
                if k=='video':
                    poster=' poster="media/'+f['poster']+'"' if f.get('poster') in lookup else '';out.append('<video class="video" controls playsinline preload="metadata" aria-label="'+title_e+'"'+poster+'><source src="'+path+'" type="video/mp4">请下载 MP4 播放。</video>')
                else:out.append('<a class="media" href="'+path+'" target="_blank" rel="noopener"><img loading="lazy" src="'+path+'" alt="'+title_e+'"></a>')
                out.append('<figcaption><h3><span class="role">'+f'{n:02d}'+'</span>'+title_e+'</h3><p>'+cap+'</p><p class="quick">');
                if k!='video':out.append('<a href="'+path+'" target="_blank" rel="noopener">放大原图</a>');
                out.append('<a href="'+path+'" download>下载 '+f['file'].split('.')[-1].upper()+' · '+sz+'</a></p></figcaption></figure>')
            else:
                out.append('<div class="row" id="'+at+'"><div><strong>'+title_e+'</strong><p>'+cap+'</p></div><div class="fileact">');
                if k=='document':out.append('<a href="'+path+'" target="_blank" rel="noopener">打开 PDF</a><br>');
                out.append('<a href="'+path+'" download>下载 · '+sz+'</a></div></div>')
        if entries:out.append('</div>');
        out.append('</section>')
    out+=['</main><footer><div>依据版本 <code>'+e(s['revision'][:12])+'</code><br>'+e(s['source_note'])+'</div><a href="#top">回到页首 ↑</a></footer>'];return shell(s['title'],''.join(out))
def catalog(rounds):
    out=[nav('../index.html','← 进度主页'),'<main id="main"><header class="hero"><p class="eyebrow">实验档案　/　组会资料</p><h1>逐轮实验记录</h1><p class="indexLead">从一个问题开始，查看当轮的模型、曲线、视频和论文页。每轮材料独立保存，也可以整包下载，离线投屏。</p><p class="meta">已归档 '+str(len(rounds))+' 轮　·　按完成时间由近到远</p></header><div class="cards">'];
    for s in rounds:
        u=s['id']+'/';url=u+'index.html';f=s['cover'];out.append('<article class="card"><div class="copy"><div class="badge">'+e(s['stage'])+'　·　'+e(s['when'][5:16].replace('T',' '))+'</div><h3><a href="'+url+'">'+e(s['title'])+'</a></h3><p>'+e(s['summary'])+'</p><p class="counts">'+e(counts(s))+'</p></div><a href="'+url+'" aria-label="查看'+e(s['title'])+'"><img class="thumb" src="'+u+'media/'+f+'" alt="" loading="lazy"></a><div class="quick"><a class="enter" href="'+url+'">打开本轮记录 →</a></div></article>');
    out.append('</div></main><footer><span>点击原图可放大；视频和 PDF 均按轮次保存。</span><a href="#top">回到页首 ↑</a></footer>');return shell('逐轮实验记录',''.join(out))
def enriched(spec,source):
    req(spec.get('schema')==SCHEMA and bool(SAFE.fullmatch(spec['id'])),'invalid schema or id'); t=dt.datetime.fromisoformat(spec['when']);req(t.utcoffset()==dt.timedelta(hours=8),'use an ISO timestamp with +08:00');req(bool(re.fullmatch('[0-9a-f]{40}',spec['revision'])),'full original Git revision required');
    for k in ('title','summary','stage','scope','source_note'):req(isinstance(spec[k],str) and bool(spec[k].strip()),'missing '+k)
    req(spec['guide'] and all(isinstance(x,str) and x for x in spec['guide']),'missing meeting notes'); seen=set()
    for f in spec['files']:
        n=f['file'];k=f['kind'];req(k in EXT and bool(FILE.fullmatch(n)) and n not in seen and Path(n).suffix in EXT[k],'invalid or duplicate asset: '+n);seen.add(n);req(f.get('title') and f.get('caption'),'title and caption required for '+n);p=source/n; req(p.resolve().parent==source.resolve() and p.is_file(),'missing source '+n);b=p.read_bytes();req(0<len(b)<=LIMIT,'invalid hosted size: '+n);f['sha256']=sha(b);f['bytes']=len(b)
    lookup={f['file']:f['kind'] for f in spec['files']};req(lookup.get(spec['cover']) in ('image','curve'),'invalid cover');
    for f in spec['files']:
        if f.get('poster'):req(f['kind']=='video' and lookup.get(f['poster']) in ('image','poster'),'invalid poster for '+f['file'])
    return spec
def putzip(z,name,data):
    info=zipfile.ZipInfo(name,(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16;z.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
def archive_root(site):return site/'archive'
def verify_round(folder):
    s=read(folder/'snapshot.json');req(s['id']==folder.name,'wrong folder identity'); expected={f['file'] for f in s['files']};req(expected=={p.name for p in (folder/'media').iterdir()},'media set differs: '+folder.name)
    for f in s['files']:
        b=(folder/'media'/f['file']).read_bytes();req(len(b)==f['bytes'] and sha(b)==f['sha256'],'media hash differs: '+f['file'])
    for k in ('page','pack'):
        entry=s['outputs'][k];req(sha((folder/entry['file']).read_bytes())==entry['sha256'],'page or pack changed: '+folder.name)
    with zipfile.ZipFile(folder/'meeting.zip') as z:
        req(z.testzip() is None and 'index.html' in z.namelist(),'invalid offline package: '+folder.name)
    return s
def all_rounds(site,check=False):
    r=archive_root(site);return sorted([(verify_round(p.parent) if check else read(p)) for p in r.glob('*/snapshot.json')],key=lambda s:(s['when'],s['id']),reverse=True)
def atomic_write(path,data):
    if path.exists() and path.read_bytes()==data:return
    f=tempfile.NamedTemporaryFile(dir=path.parent,delete=False);tmp=Path(f.name)
    try:
        with f:f.write(data)
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()
H0='<!-- meeting-archive:start -->'
H1='<!-- meeting-archive:end -->'
def home_entry(ss):
    last=ss[0];return H0+'<nav id="meeting-archive" aria-label="组会归档" style="margin:20px 0;padding:16px 20px;border-left:4px solid #246a65;background:#eaf3f1"><strong style="font-size:20px;color:#173d42">组会归档</strong><p style="margin:3px 0 9px">每轮独立保存原图、曲线、视频和论文页，可离线投屏。</p><a style="font-weight:650;margin-right:1.5rem" href="archive/index.html">浏览全部 '+str(len(ss))+' 轮 →</a><a href="archive/'+last['id']+'/index.html">最近一轮：'+e(last['title'])+'</a></nav>'+H1
def sync_home(site,ss):
    home=site/'index.html'
    if not home.is_file() or not ss:return
    s=home.read_text(encoding='utf-8');entry=home_entry(ss)
    if H0 in s and H1 in s:s=s[:s.index(H0)]+entry+s[s.index(H1)+len(H1):]
    elif '<!-- meeting-archive-slot -->' in s:s=s.replace('<!-- meeting-archive-slot -->',entry,1)
    elif '<h1>论文修改与仿真观察</h1>' in s and '</small>' in s:s=s.replace('</small>','</small>'+entry,1)
    else:return
    atomic_write(home,s.encode())
def index(site):
    r=archive_root(site);r.mkdir(parents=True,exist_ok=True);ss=all_rounds(site);atomic_write(r/'index.html',catalog(ss).encode());sync_home(site,ss);return len(ss)
def add(site,spec,src):
    s=enriched(read(spec),src);r=archive_root(site);r.mkdir(exist_ok=True,parents=True); dest=r/s['id'];
    if dest.exists():
        old=verify_round(dest);old.pop('outputs');req(same_json(old,s),'ROUND IS FROZEN: use a new id for any revision, including captions');index(site);return 'unchanged: '+s['id']
    # Prepare outside the public tree and rename only when media, HTML, hashes and the offline package all exist.
    with tempfile.TemporaryDirectory(dir=r.parent.parent,prefix='.prepare-round-') as tmp:
        temp=Path(tmp)/s['id'];media=temp/'media';media.mkdir(parents=True)
        for f in s['files']:
            p=media/f['file'];shutil.copyfile(src/f['file'],p);req(sha(p.read_bytes())==f['sha256'],'source changed while copying '+p.name)
        (temp/'index.html').write_bytes(detail(s).encode()); omitted={f['file'] for f in s['files'] if f.get('separate',False)}
        with zipfile.ZipFile(temp/'meeting.zip','w') as z:
            putzip(z,'index.html',detail(s,True,omitted).encode());putzip(z,'files.json',jbytes({'round':s['id'],'revision':s['revision'],'files':[f for f in s['files'] if f['file'] not in omitted],'separate_files':sorted(omitted)}));
            for f in s['files']:
                if f['file'] not in omitted:putzip(z,'media/'+f['file'],(media/f['file']).read_bytes())
        req((temp/'meeting.zip').stat().st_size<=LIMIT,'offline pack too large: mark a large attachment separate in the spec');s['outputs']={k:{'file':n,'sha256':sha((temp/n).read_bytes())} for k,n in [('page','index.html'),('pack','meeting.zip')]};(temp/'snapshot.json').write_bytes(jbytes(s));verify_round(temp);temp.rename(dest)
    index(site);return 'added: '+s['id']
def main():
    par=argparse.ArgumentParser(description=__doc__);sub=par.add_subparsers(dest='command',required=True)
    for name in ('add','verify','index'):
        p=sub.add_parser(name);p.add_argument('--site',type=Path,required=True,help='Static site root containing homepage; archive/ is created inside it')
        if name=='add':p.add_argument('--spec',required=True,type=Path);p.add_argument('--source',required=True,type=Path,help='Directory with the individually listed source assets')
    opt=par.parse_args();site=opt.site.resolve()
    if opt.command=='add':print(add(site,opt.spec,opt.source))
    elif opt.command=='index':print('indexed:',index(site))
    else:
        ss=all_rounds(site,True);req(bool(ss),'no records');req((archive_root(site)/'index.html').read_bytes()==catalog(ss).encode(),'archive index differs');print('verified:',len(ss),'rounds /',sum(len(s['files']) for s in ss),'original and derived files')
if __name__=='__main__':main()
