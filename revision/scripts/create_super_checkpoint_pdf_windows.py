from pathlib import Path
import json, argparse
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
import pypdfium2 as pdfium

parser=argparse.ArgumentParser(description='Build the Windows stage PDF from the local artifact workspace.')
parser.add_argument('--artifact-root',type=Path,default=Path(__file__).resolve().parent)
ROOT=parser.parse_args().artifact_root.resolve();DATA=ROOT/'SUPER平台统一'
OUT=ROOT/'阶段C_SUPER平台与接口检查预览.pdf'
pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/Deng.ttf'))
pdfmetrics.registerFont(TTFont('CNB','C:/Windows/Fonts/Dengb.ttf'))
pdfmetrics.registerFont(TTFont('TR','C:/Windows/Fonts/times.ttf'))
pdfmetrics.registerFont(TTFont('TB','C:/Windows/Fonts/timesbd.ttf'))
c=canvas.Canvas(str(OUT),pagesize=(595.276,841.89));c.setTitle('SUPER平台、相机与SDK模型检查')
def text(s,x,y,size=10,font='CN',color='#283239'):
    c.setFont(font,size);c.setFillColor(HexColor(color));c.drawString(x,y,s)
def rule(y):
    c.setStrokeColor(HexColor('#bac2c7'));c.setLineWidth(.45);c.line(38,y,557,y)
def para(s,x,y,w=519,size=9.6,leading=15,color='#283239'):
    p=Paragraph(s,ParagraphStyle('p',fontName='CN',fontSize=size,leading=leading,textColor=HexColor(color),wordWrap='CJK'))
    _,h=p.wrap(w,800);p.drawOn(c,x,y-h);return y-h
def image(path,x,y,w,h,crop=None):
    src=ImageReader(str(path));iw,ih=src.getSize()
    if crop is None:c.drawImage(src,x,y,w,h,preserveAspectRatio=True,anchor='c');return
    # Page-layout clipping only. The source raster is unchanged.
    l,t,r,b=crop;scale=min(w/(r-l),h/(b-t));dw=(r-l)*scale;dh=(b-t)*scale
    xx=x+(w-dw)/2;yy=y+(h-dh)/2
    c.saveState();p=c.beginPath();p.rect(xx,yy,dw,dh);c.clipPath(p,stroke=0,fill=0)
    c.drawImage(src,xx-l*scale,yy-(ih-b)*scale,iw*scale,ih*scale);c.restoreState()
def footer(page):
    rule(58);text('2026-11-17  |  SUPER + Qingyu development configuration',38,39,8,'TR')
    text(str(page),548,39,9,'TR')

text('SUPER 平台与传感器装配',38,800,19,'CNB')
text('按本机总装定位机身相机，恢复 Qingyu SDK 臂体与完整夹爪',38,777,10.5)
rule(762)
text('a   Registered platform',38,742,11,'TB')
image(DATA/'super_backend/geometry/perspective_cad.png',38,365,305,368,(135,55,710,735))
text('b   Native camera registration',350,742,11,'TB')
image(DATA/'super_backend/geometry/body_camera_closeup.png',350,536,207,198,(160,160,710,660))
text('c   SDK gripper and distal camera',350,515,11,'TB')
image(DATA/'super_backend/geometry/distal_closeup.png',350,353,207,153,(110,100,730,635))
rule(346)
text('模型来源与几何参数',38,324,12,'CNB')
rows=[('机架来源','SUPER-Hardware 原始 SolidWorks / STEP，31个文件校验一致'),
      ('旋翼位置','配准后旋翼轴：x / y 约 ±99.0 mm；分配矩阵读取模型位置'),
      ('长起落架','115个既有原生部件；保留等比例尺寸，使用独立碰撞代理'),
      ('刚性包络','高度 494.48 mm；相对 UAV 原点 z ∈ [-374.03, 120.45] mm'),
      ('臂体与夹爪','臂体使用SDK的15个球标记；夹爪恢复完整原始STL表面')]
for i,(a,b) in enumerate(rows):
    y=296-i*29;text(a,38,y,9.5,'CNB');text(b,119,y,9.1);rule(y-11)
para('机身相机相对MID360的位置由原生总装刚性配准得到。末端相机随工具固定转动。中央转接件、精确传感器标定及动力学参数继续作为待验证项；原始模型与失败日志均保留。',38,132,size=9.4,leading=15)
text('机架原始来源：github.com/hku-mars/SUPER-Hardware',38,76,8.7,'CN')
c.linkURL('https://github.com/hku-mars/SUPER-Hardware',(38,73,370,87),relative=0)
footer(1);c.showPage()

text('接口、取景与积分状态检查',38,800,19,'CNB')
text('本页数值来自新版模型；作为开发验证记录，尚未计入穿门任务统计',38,777,10)
rule(762)
image(DATA/'super_backend_checks.png',32,528,531,219)
fixed=json.loads((DATA/'super_backend/fixed/manifest.json').read_text())
hover=json.loads((DATA/'super_backend/hover/manifest.json').read_text())
peak=max(fixed['peak_total_curvature_deg'],hover['peak_total_curvature_deg'])
para('固定机身 / 悬停各积分14 s，最大总曲率 %.3f°，0碰撞、0次超过120°。悬停采用内部控制器及真值反馈，位置偏差峰值 %.2f mm。48个形态样本中18个自碰撞，说明角度限幅还必须配合整段过渡检查。'%(peak,hover['max_position_error_m']*1000),38,527,size=9.4,leading=15)
text('c   Body camera',38,454,11,'TB');text('d   Distal camera',308,454,11,'TB')
image(DATA/'camera_alignment/navigation.png',38,270,249,177)
image(DATA/'camera_alignment/wrist_rgbd.png',308,270,249,177)
para('两图直接取自 MuJoCo 的相机渲染：红色目标在上、蓝色在左，镜头前方目标可见。该检查确认外观正面、仿真光轴及图像上下方向一致。',38,255,size=9.4,leading=15)
rule(215)
text('已完成与下一步',38,194,12,'CNB')
para('已完成：7条真实SDK隔离ROS指令校验；10项行为测试；两路相机取景与完整夹爪几何检查；前向弯臂固定 / 悬停积分。<br/>同时修正SDK方位转换：SDK角度 = 90° - PCC方位角。本轮交付后暂停，待评审后再进入规划、飞控和感知链路。',38,176,size=9.5,leading=16)
para('先确认平台和观测关系，再评价规划与控制效果。这样新的穿门图、净空曲线与日志才能对应同一台机器人。',38,103,size=9.6,leading=15,color='#315d6d')
footer(2);c.save()
doc=pdfium.PdfDocument(str(OUT))
for i in range(len(doc)):doc[i].render(scale=1.6).to_pil().save(ROOT/f'super_checkpoint_page_{i+1}.png')
print(OUT)
