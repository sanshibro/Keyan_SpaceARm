"""Prepare the single-gate progress view in the existing group site checkout."""
import argparse
from pathlib import Path
from PIL import Image
import json
import hashlib
import shutil
import subprocess
from tempfile import TemporaryDirectory

BRANCH='feat/paper-mechanism-rolling-20261117'
HOST='https://raw.githubusercontent.com/sanshibro/Keyan_SpaceARm/'+BRANCH+'/revision/data/round_gate_feedback/'
PAGE="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>单门闭环飞行记录 · 空中连续体研究组</title>
<style>
:root{color-scheme:light;--ink:#20343d;--muted:#546a72;--rule:#dce4e5;--accent:#217669}
*{box-sizing:border-box}body{margin:0;color:var(--ink);font:15px/1.64 "Microsoft YaHei",Arial,sans-serif;background:#fff}
main{max-width:1040px;margin:auto;padding:26px 30px 64px}header{border-bottom:2px solid var(--ink);padding-bottom:18px}
nav a,a{color:#165f69;text-underline-offset:3px}h1{font:600 clamp(25px,4vw,40px)/1.25 "Songti SC","SimSun",serif;margin:18px 0 10px}
h2{font:600 22px/1.35 "Songti SC","SimSun",serif;margin:32px 0 10px}p{margin:6px 0}small,.muted{color:var(--muted)}
.head{display:flex;gap:24px;align-items:end;justify-content:space-between}.num{font:500 34px/1.1 Georgia,serif;min-width:118px;text-align:right}
.num small{display:block;font:13px/1.6 Arial,sans-serif}.bar{height:4px;background:var(--rule);margin:16px 0 0}.bar i{display:block;height:100%;background:var(--accent)}
figure{margin:0}figure img{max-width:100%;height:auto;display:block}figcaption{color:var(--muted);font-size:13px}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap}th{color:#46595f;background:#f5f7f7}td:first-child,th:first-child{text-align:left}
fieldset{border:0;padding:0;display:flex;gap:18px;flex-wrap:wrap;margin:8px 0}select{padding:5px;border:1px solid #bdcccc;background:white;font:inherit;color:var(--ink)}
details{margin-top:24px;border-top:1px solid var(--rule);padding-top:12px}summary{cursor:pointer;color:var(--accent)}footer{margin-top:22px;color:var(--muted);font-size:13px}
@media(max-width:560px){main{padding:18px 14px 38px}.num{font-size:23px;min-width:82px}.head{gap:6px}th,td{padding:6px}}
</style>
<main><nav><a href="../index.html">组会首页</a>　/　<a href="../archive/index.html">轮次归档</a></nav>
<header><div class="head"><div><h1>受限单门通过与斜向对准</h1><p>PX4 自由飞行 · MuJoCo 柔性臂 · 原生腕部 RGB-D SLAM</p></div><div class="num"><b id="n">—</b> / 60<small>已独立评分的飞行</small></div></div><div class="bar"><i id="progress"></i></div></header>
<p class="muted" id="status" role="status">正在读取研究分支的阶段记录。</p>
<h2>配对飞行比较</h2><p>每格三趟。通过要求机身与臂爪净空均达到 6.0 cm，越门后工具在 3.5 cm 和 3.0° 以内连续保持 1 秒。</p>
<div class="scroll"><table><thead><tr><th>反馈源</th><th>迟滞 · 清晰</th><th>迟滞 · 遮挡</th><th>差模 · 清晰</th><th>差模 · 遮挡</th></tr></thead><tbody id="summary"></tbody></table></div>
<small>格内为通过数 / 已评分数。挡板在飞行第 28–32 秒覆盖腕部视野。</small>
<h2>腕部失视时的真实变形</h2><figure><a href="wrist-window.webp"><img src="wrist-window.webp" width="900" alt="原生腕部 RGB-D 三帧与同步的真实弯曲角和方位角"></a><figcaption>种子 26112131，进门前差模偏移。三帧分别取自视野清晰、完全遮挡和视觉恢复的时刻。灰带内的虚线给出这趟融合飞行中保留最后视觉观测的结果。</figcaption></figure>
<h2>逐趟记录</h2><fieldset><label>材料　<select id="material"><option value="">全部</option><option value="nominal">基础迟滞</option><option value="differential">迟滞与差模偏移</option></select></label><label>视野　<select id="view"><option value="">全部</option><option value="natural">清晰</option><option value="occluded">动态遮挡</option></select></label></fieldset>
<div class="scroll"><table><thead><tr><th>种子 / 方法</th><th>工况</th><th>评价</th><th>臂爪净空</th><th>保护停顿</th><th>原始记录</th></tr></thead><tbody id="list"></tbody></table></div>
<details><summary>查看开发种子的真值参考飞行</summary><figure><img src="qualification.webp" width="740" alt="真值反馈下的实际弯角、方位角和工具误差曲线"><figcaption>开发种子 26112090；灰带标出预定的四秒动态遮挡时段。</figcaption></figure></details>
<footer>配对种子 26112131、26112137、26112143。原始包收录 PX4 ULog、ROS bag、物理轨迹、传感器记录与 SHA-256。</footer></main>
<script>
const host="__HOST__", names={vision:"纯视觉",strain:"纯应变",fixed:"固定噪声 KF",proposed:"增广融合",oracle:"真值参考"};
const conditions=[["nominal","natural"],["nominal","occluded"],["differential","natural"],["differential","occluded"]];
const el=id=>document.getElementById(id);let data={cases:{},evidence:{}};
function cell(tr,value){const c=document.createElement("td");c.textContent=value;tr.append(c);return c}
function render(){const rows=Object.values(data.cases);el("n").textContent=rows.length;el("progress").style.width=(100*rows.length/60)+"%";
const group=el("summary");group.replaceChildren();for(const [method,label] of Object.entries(names)){let tr=document.createElement("tr");cell(tr,label);for(const [mat,view] of conditions){let r=rows.filter(x=>x.case.method===method&&x.case.material===mat&&x.case.vision===view);cell(tr,r.length?r.filter(x=>x.success).length+" / "+r.length:"—")}group.append(tr)}
const body=el("list");body.replaceChildren();const picked=rows.filter(x=>(!el("material").value||x.case.material===el("material").value)&&(!el("view").value||x.case.vision===el("view").value));
for(const x of picked){const c=x.case,tr=document.createElement("tr");cell(tr,c.seed+" / "+names[c.method]);cell(tr,(c.material==="nominal"?"迟滞":"差模")+" · "+(c.vision==="natural"?"清晰":"遮挡"));let outcome=x.success?"通过":!x.crossed?"门前保持":!x.safe&&!x.dwell_reached?"安全与对准未达标":!x.safe?"已对准 · 安全未达标":"已越门 · 对准未达标";cell(tr,outcome);cell(tr,(100*x.arm_tool_min_m).toFixed(2)+" cm");cell(tr,x.hold_s.toFixed(2)+" s");const v=data.evidence[c.case],td=cell(tr,"");if(v){let a=document.createElement("a");a.textContent="下载 ZIP";a.href=host+"evidence/"+encodeURIComponent(v.file);td.append(a)}body.append(tr)}
if(!picked.length){let tr=document.createElement("tr");cell(tr,"等待该工况的评分记录").colSpan=6;body.append(tr)}}
async function update(){try{let r=await fetch(host+"decisions/metrics.json?stage="+Math.floor(Date.now()/60000),{cache:"no-store"});if(!r.ok)throw Error(String(r.status));let v=await r.json();if(!v.cases||!v.evidence)throw Error("schema");if(Object.keys(v.cases).length>=Object.keys(data.cases).length)data=v;el("status").textContent=Object.keys(data.cases).length===60?"60 趟飞行已评分。":"研究分支每完成五趟发布一次记录，页面每分钟刷新。"}catch(e){el("status").textContent=Object.keys(data.cases).length?"已读取组会站内存档，可查看逐趟成绩。":"正在读取组会存档。"}render()}
async function initialize(){try{let r=await fetch("checkpoint.json");if(r.ok){data=await r.json();render();el("status").textContent="已读取组会站内存档。"}}catch(e){}await update()}
el("material").onchange=render;el("view").onchange=render;render();initialize();setInterval(update,60000);
</script></html>"""

def restore_public_media(stage):
    """Keep historic root bookmarks working and retain byte-identical archive originals."""
    from pypdf import PdfReader,PdfWriter
    from pypdf.filters import ASCII85Decode
    from pypdf.generic import NameObject
    dist=stage/'site-cache/dist'
    main=dist/'model_paper_page.pdf'
    original=dist/'archive/2026-11-17-c2/media/paper.pdf'
    qa=stage/'qa'/'historic-media.json'
    art=main.read_bytes()
    orig_sha=hashlib.sha256(original.read_bytes()).hexdigest()
    if hashlib.sha256(art).hexdigest()==orig_sha:
        writer=PdfWriter(clone_from=str(original))
        changed=0
        for o in writer._objects:
            if not hasattr(o,'_data') or o.get('/Filter')!=['/ASCII85Decode','/FlateDecode'] or o.get('/DecodeParms') is not None:
                continue
            # Drop only ASCII transport encoding. The compressed image payload is identical.
            o._data=ASCII85Decode.decode(o._data)
            o[NameObject('/Filter')]=NameObject('/FlateDecode')
            changed+=1
        if not changed:raise ValueError('Historic PDF contains no recognized transport encoding')
        with TemporaryDirectory(dir=stage/'qa') as tmp:
            candidate=Path(tmp)/'compacted.pdf'
            writer.write(candidate)
            pages=len(PdfReader(str(original)).pages)
            if len(PdfReader(str(candidate)).pages)!=pages:raise ValueError('Historic page count changed')
            pixel_hash=[]
            for i in range(1,pages+1):
                args=['pdftoppm','-r','144','-f',str(i),'-l',str(i),'-singlefile']
                rasters=[subprocess.check_output(args+[str(p)]) for p in (original,candidate)]
                if any(not x.startswith(b'P6') or len(x)<100000 for x in rasters):raise IOError('Historic render missing')
                pixels=[hashlib.sha256(x).hexdigest() for x in rasters]
                if pixels[0]!=pixels[1]:raise ValueError('Historic page pixels changed')
                pixel_hash.append(pixels[0])
            result=candidate.read_bytes()
            if len(result)>=len(art)-1024*1024:raise ValueError('Insufficient room for the group archive')
            main.write_bytes(result)
            qa.write_text(json.dumps(dict(original_sha256=orig_sha,web_sha256=hashlib.sha256(result).hexdigest(),
                        matched_page_rasters_144dpi=pixel_hash,original_bytes=len(art),web_bytes=len(result)),indent=2)+'\n',encoding='utf8')
    elif not qa.is_file() or json.loads(qa.read_bytes())['web_sha256']!=hashlib.sha256(art).hexdigest():
        raise ValueError('Unrecognized live PDF revision')
    aliases={'super_hover_review.mp4':'archive/2026-11-17-c2/media/hover.mp4',
             'model_checkpoint.pdf':'archive/2026-11-17-c2/media/review.pdf',
             'replay.json':'archive/2026-11-17-a/media/replay.json'}
    redirects=dist/'_redirects'
    lines=redirects.read_text(encoding='utf-8').splitlines()
    for name,archive in aliases.items():
        current=dist/name;canonical=dist/archive
        if not current.is_file():shutil.copyfile(canonical,current)
        if current.read_bytes()!=canonical.read_bytes():raise ValueError('Historic file differs: '+name)
        lines=[row for row in lines if row!=('/'+name+' /'+archive+' 302')]
    redirects.write_bytes(('\n'.join(lines)+'\n').encode('utf8'))

def build(stage):
    dist=stage/'site-cache/dist'
    dest=dist/'gate-live';dest.mkdir(exist_ok=True)
    (dest/'index.html').write_text(PAGE.replace('__HOST__',HOST),encoding='utf-8')
    metric=json.loads((stage/'decisions/metrics.json').read_bytes())
    keep=('case','success','safe','crossed','dwell_reached','arm_tool_min_m','hold_s')
    # Every five formally scored flights are committed and made publicly downloadable.
    public_count=len(metric['cases'])//5*5
    public_cases=list(metric['cases'])[:public_count]
    selected={k:{key:metric['cases'][k][key] for key in keep} for k in public_cases}
    (dest/'checkpoint.json').write_text(json.dumps(dict(cases=selected,evidence={k:metric['evidence'][k] for k in selected}),ensure_ascii=False,separators=(',',':')),encoding='utf8')
    shutil.copyfile(stage/'review/gate_wrist_window.webp',dest/'wrist-window.webp')
    with Image.open(stage/'pilot/两自由度真值参考飞行.png') as im:
        im.thumbnail((820,750));im.convert('RGB').save(dest/'qualification.webp','WEBP',quality=78)
    home=dist/'index.html';s=home.read_text(encoding='utf-8')
    if 'href="gate-live/index.html"' not in s:
        anchor='<section id="reviewVideo">'
        if s.count(anchor)!=1:raise ValueError('Cannot identify home review section')
        link='<section><h2>单门闭环飞行</h2><p><a href="gate-live/index.html">查看五种反馈源的配对飞行、净空和逐趟日志</a></p></section>'
        home.write_bytes(s.replace(anchor,link+anchor,1).encode('utf-8'))
    restore_public_media(stage)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',type=Path);a=p.parse_args();build(a.stage)
