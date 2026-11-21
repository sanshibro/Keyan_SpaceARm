"""Mirror the registered NUC run into a self-refreshing, local review page."""
import base64,html,json,os,subprocess,time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REMOTE="""import base64, json
from pathlib import Path
D=Path('/home/uav/paper_revision_20261117/revision/data/round_m1_coupled/publication')
cells=[]
for path in sorted(D.glob('*/plan/decision.json')):
    case=path.parents[1];f=case/'flight';state={'name':case.name,'stage':'queued'}
    if (f/'progress.json').is_file():
        h=json.loads((f/'progress.json').read_text());state.update(stage='flying',time=h['sim_s'],phase=h['phase'],theta=h['bend_deg'],xyz=h['world_relative_enu_m'])
        if (f/'control-summary.json').is_file():state['stage']='logged'
        if (case/'audit/metrics.json').is_file():state.update(stage='scored',passed=json.loads((case/'audit/metrics.json').read_text())['task_success'])
        if state['stage']=='flying' and (f/'optics/index.jsonl').exists():
            try:
                with (f/'optics/index.jsonl').open('rb') as stream:
                    stream.seek(max(0,(f/'optics/index.jsonl').stat().st_size-3900));lines=stream.read().splitlines()
                for line in reversed(lines):
                    try:o=json.loads(line);jpeg=(f/'optics'/o['rgb']).read_bytes()
                    except (OSError,ValueError,KeyError):continue
                    if jpeg.endswith(bytes([255,217])):
                        state.update(image=base64.b64encode(jpeg).decode(),exposure=o['t']);break
            except OSError:pass
    cells.append(state)
print(json.dumps(cells))
"""
CSS="""*{box-sizing:border-box}body{background:#f7f8f5;color:#243c36;font:16px/1.55 'Microsoft YaHei',sans-serif;max-width:1060px;margin:auto;padding:26px}h1,h2{font-family:Georgia,SimSun,serif}h1{font-weight:500;margin:0 0 6px}p{margin:5px 0 18px}#cams{display:flex;gap:18px;flex-wrap:wrap}figure{margin:12px 0;flex:1;min-width:260px}img{width:100%;max-width:430px}table{width:100%;border-collapse:collapse;background:white}td,th{text-align:left;border-bottom:1px solid #d6dfd8;padding:7px 9px}.run{color:#006c61;font-weight:650}small{color:#586d61}"""
NAMES={'straight':'直臂','fixed':'固定形态','sequential':'先机身后机械臂','unregularized':'联合几何 / 时序消融','coupled':'完整六维联合'}
STATUS={'queued':'等待飞行','flying':'飞行中','logged':'已落盘','scored':'已评分'}
def once():
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=6','nuc-uav','python3','-'],input=REMOTE,capture_output=True,text=True,encoding='utf8',timeout=16)
    r.check_returncode();cells=json.loads(r.stdout);now=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cams=[];rows=[]
    cells.sort(key=lambda c:(c['name'].split('-')[1],list(NAMES).index(c['name'].split('-')[0])))
    for c in cells:
        m,seed=c['name'].rsplit('-',1);state=c['stage']
        if 'image' in c:cams.append(f"<figure><img alt='腕部原始曝光' src='data:image/jpeg;base64,{c['image']}'><figcaption>{NAMES[m]} · {seed} · 曝光 {c['exposure']:.2f} s</figcaption></figure>")
        shown=STATUS[state];shown+=(' · 达标' if c['passed'] else ' · 未达整趟阈值') if 'passed' in c else ''
        rows.append(f"<tr><td>{seed}</td><td>{NAMES[m]}</td><td class='{'run' if state=='flying' else ''}'>{shown}</td><td>{c.get('phase','—')}</td><td>{c.get('theta','—')}</td></tr>")
    n=sum(c['stage'] in ('logged','scored') for c in cells)
    h=f"<!doctype html><html lang='zh'><meta charset='utf-8'><meta http-equiv='refresh' content='18'><meta name='viewport' content='width=device-width,initial-scale=1'><title>M1 配对飞行现场</title><style>{CSS}</style><h1>三关管廊 · 配对飞行现场</h1><p>已记录 {n} / 15 趟　<small>NUC 最近采样 {now}</small></p><section id='cams'>{''.join(cams)}</section><table><tr><th>配对种子</th><th>规划策略</th><th>状态</th><th>控制阶段</th><th>弯曲幅值 / °</th></tr>{''.join(rows)}</table></html>"
    out=ROOT/'review/M1_飞行现场.html';tmp=out.with_suffix('.tmp');tmp.write_text(h,'utf8');os.replace(tmp,out)
    return len(cells)==15 and all(c['stage']=='scored' for c in cells)
if __name__=='__main__':
    for _ in range(800):
        try:
            if once():break
        except (OSError,ValueError,subprocess.SubprocessError):pass
        time.sleep(18)
