"""Standalone two-page Chinese visual-stage review; no flight data is consumed."""
import json
from pathlib import Path
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3,landscape
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

STAGE=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def make(destination=None):
    qa=STAGE/'qa/static';cert=read(qa/'certified.json');scene=read(STAGE/'scene/manifest.json');opt=read(qa/'flange/flange-pixel-audit.json')
    if not (cert['scene_sha256']==scene['scene_sha256']==opt['scene']):raise ValueError('Visual-stage evidence is not synchronized')
    out=Path(destination or STAGE/'review/视觉阶段定稿_三关几何与同源光学_双页.pdf');out.parent.mkdir(parents=True,exist_ok=True)
    try:
        pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/simsun.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('CNB','C:/Windows/Fonts/simhei.ttf'))
    except (OSError,ValueError):
        pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'));pdfmetrics.registerFontFamily('CN',normal='STSong-Light');pdfmetrics.registerFontFamily('CNB',normal='STSong-Light')
    cn='CN' if 'CN' in pdfmetrics.getRegisteredFontNames() else 'STSong-Light';bold='CNB' if 'CNB' in pdfmetrics.getRegisteredFontNames() else cn
    W,H=landscape(A3);c=canvas.Canvas(str(out),pagesize=(W,H));c.setTitle('视觉阶段定稿：三关工业几何与同源光学')
    def txt(x,y,s,size=13,color='#253e48',heavy=False):
        c.setFont(bold if heavy else cn,size);c.setFillColor(HexColor(color));c.drawString(x,y,s)
    def para(x,y,w,s,size=13,leading=21,color='#364d55'):
        style=ParagraphStyle('p',fontName=cn,fontSize=size,leading=leading,textColor=HexColor(color),wordWrap='CJK')
        p=Paragraph(s,style);ww,hh=p.wrap(w,H);p.drawOn(c,x,y-hh);return y-hh
    def picture(path,x,y,w,h):
        with Image.open(path) as im:iw,ih=im.size
        scale=min(w/iw,h/ih);ww,hh=iw*scale,ih*scale;c.drawImage(ImageReader(str(path)),x+(w-ww)/2,y+(h-hh)/2,ww,hh,mask='auto')
    def rule(x,y,w):c.setStrokeColor(HexColor('#a7b9be'));c.setLineWidth(.6);c.line(x,y,x+w,y)
    def header(n,title):
        txt(32,H-34,'M2  /  视觉阶段定稿',12,'#58747b');txt(32,H-70,title,27,heavy=True);rule(32,H-86,W-64)
        txt(32,23,'静态几何与原生相机证据  ·  21 趟正式飞行结果另册记录',10,'#5f747b');txt(W-44,23,str(n),11)
    header(1,'三关连续长廊：可见制造结构与净空逐项对应')
    picture(qa/'three-gate-static-review.png',28,65,790,H-165)
    x=845;w=W-x-32;y=H-115
    txt(x,y,'三种连续形态约束',18,heavy=True);y-=24
    y=para(x,y,w,'G1 高位窄门：前折机械臂，抬高机身通过下槛。<br/>G2 压顶风管与错位支管：机身下降 0.32 m，并转为侧弯。<br/>G3 反向偏置窄门：再切回前折，进入倾斜法兰检视区。',13,22)
    y-=23;txt(x,y,'冻结几何与制造表达',18,heavy=True);y-=25
    y=para(x,y,w,'154 个共享 UV 网格、626 个实体细节；45 个碰撞体逐字段保持不变。两门净开口均为 0.86 × 0.74 m。<br/>倒角薄框、沉孔、百叶、凹刻管箍与 100 mm 甲板格缝均位于已注册包络内，不占用净空。',13,21)
    y-=22;txt(x,y,'独立静态距离下界 / mm',18,heavy=True);y-=27
    cols=[x,x+69,x+141,x+226]
    for xx,s in zip(cols,['关卡','外障','危险自碰','配合']):txt(xx,y,s,12,heavy=True)
    rule(x,y-10,w);y-=32
    for g in cert['gate_witnesses']:
        values=[g['gate'],'%.2f'%(1000*g['external_lower_m']),'%.2f'%(1000*g['hazard_lower_m']),'%.3f'%(1000*g['fit_lower_m'])]
        for xx,v in zip(cols,values):txt(xx,y,v,14)
        y-=27
    y-=3;y=para(x,y,w,'三关均满足外障 ≥ 105 mm、危险自碰 ≥ 15 mm，注册配合对保持零穿透。支腿与所有非连接部件的碰撞对完整保留。',12,19)
    y-=22;txt(x,y,'原生启动构型至悬停',18,heavy=True);y-=25
    launch=cert['launch_preflight']
    y=para(x,y,w,'低入口甲板顶面 0.55 m；主甲板从 x = 0.60 m 起抬高至 1.25 m。<br/>75 个平移采样均无物理接触；连续平移包络的外障保守下界为 %.2f mm。'%(1000*launch['interpolation_external_lower_m']),13,21)
    rule(32,48,W-64);txt(32,36,'场景 SHA-256：'+scene['scene_sha256'],9,'#647b81')
    c.showPage();header(2,'同源腕部光学：靶心、刻度与八个沉孔均可辨识')
    picture(qa/'flange/flange-oblique.png',32,365,690,345)
    picture(qa/'flange/flange-wrist.png',752,410,W-784,300)
    txt(34,728,'a  外部法兰机位：相机、机械臂与工具的真实相对位置',14,heavy=True)
    txt(752,728,'b  原生 wrist_rgbd：480 × 360 像素',14,heavy=True)
    txt(752,385,'镜头中心深度：%.3f mm'%(1000*opt['center']['actual_depth_m']),17,heavy=True)
    rule(32,348,W-64)
    x1,x2,x3=32,430,825;w1,w2,w3=365,365,W-857
    txt(x1,318,'01  同一套可见实体',18,heavy=True)
    para(x1,289,w1,'腕部与外部相机读取同一工业网格与 M1 刷纹 UV。白色背景及近侧透明剖切梁对所有相机一致；物理接触体始终保留。',14,23)
    para(x1,205,w1,'三关图中的三架 UAV 表示三个分别核验的静态姿态；图中没有合成飞行轨迹或飞行成绩。',13,22)
    txt(x2,318,'02  可见制造细节',18,heavy=True)
    para(x2,289,w2,'法兰含 79 个实体件：分层车削壳体、铜色双环、银色量规环、微凹靶心、36 个刻度，以及 8 个与物理注册位置一致的六角沉孔。',14,23)
    para(x2,184,w2,'斜撑、管箍与沉孔均收在既有圆柱包络内。门框倒角与表面压纹收在门板包络内，内壁开口尺寸保持不变。',13,22)
    txt(x3,318,'03  实测光学核查',18,heavy=True)
    values=[('注册工作距离','%.3f mm'%(1000*opt['stand_off_to_datum_m'])),
      ('靶心原生深度','%.3f mm'%(1000*opt['center']['actual_depth_m'])),
      ('沉孔深度核验','%d / %d'%(opt['socket_centers_depth_verified'],len(opt['bolts']))),('法兰可见像素',str(opt['flange_pixels'])),('铜色细节像素',str(opt['copper_pixels']))]
    yy=283
    for label,value in values:
        txt(x3,yy,label,13);txt(x3+151,yy,value,14,'#286a75');rule(x3,yy-11,w3);yy-=37
    para(32,110,W-64,'核查来源：冻结场景 manifest、独立凸距离 certified.json、75 点 launch-clearance.json，以及基于原生像素/深度/几何 ID 的 flange-pixel-audit.json。',11,17)
    rule(32,48,W-64);txt(32,36,'物理几何 SHA-256：'+scene['collision_geometry_sha256'],9,'#647b81')
    c.save();return out
if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path);v=a.parse_args();print(make(v.output))
