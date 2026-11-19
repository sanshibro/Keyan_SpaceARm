"""Make the two-page meeting proof from scored flights and camera-ready figure plates."""
import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

W,H=630,816
INK='#233b43';MUTED='#57696f';LINE='#b7c6c6';TEAL='#286d63'
fontroot=Path(os.environ.get('G1_PDF_FONTS','C:/Windows/Fonts'))
for tag,name in [('CN','msyh.ttc'),('B','msyhbd.ttc')]:
    pdfmetrics.registerFont(TTFont(tag,str(fontroot/name)))


def text(c,x,y,s,size=10,font='CN',color=INK,align='left'):
    c.setFillColor(HexColor(color));c.setFont(font,size)
    {'left':c.drawString,'center':c.drawCentredString,'right':c.drawRightString}[align](x,H-y,s)


def line(c,x,y,x2,y2,w=.6,color=LINE):
    c.setStrokeColor(HexColor(color));c.setLineWidth(w);c.line(x,H-y,x2,H-y2)


def image(c,path,x,y,w,h):
    c.drawImage(ImageReader(str(path)),x,H-y-h,w,h,mask='auto',preserveAspectRatio=True,anchor='c')


def header(c,n,title,sub):
    text(c,35,26,'SUPER  /  QINGYU     单门闭环飞行',8.4,color=MUTED)
    text(c,595,26,'G1    /    2026.11.19',8.4,color=MUTED,align='right')
    line(c,35,38,595,38,.9,INK)
    text(c,35,80,title,23,'B')
    text(c,36,104,sub,9.9,color=MUTED)
    line(c,35,777,595,777)
    text(c,35,796,'组会审阅    /    已知地图 · 空载标称模型 · 软件在环',8,color=MUTED)
    text(c,595,796,'%02d   /   02'%n,8,color=MUTED,align='right')


def para(c,y,lines):
    for j,s in enumerate(lines):
        text(c,36,y+17*j,s,9.7)


def table(c,top,head,rows,widths):
    xx=[36]
    for v in widths:xx.append(xx[-1]+v)
    assert xx[-1]<=596
    line(c,36,top,xx[-1],top,.85,INK)
    for j,s in enumerate(head):text(c,(xx[j]+xx[j+1])/2,top+15,s,8.5,'B',align='center')
    line(c,36,top+23,xx[-1],top+23)
    for i,r in enumerate(rows):
        for j,s in enumerate(r):
            text(c,(xx[j]+xx[j+1])/2,top+39+22*i,s,9.2,align='center')
    line(c,36,top+47+22*(len(rows)-1),xx[-1],top+47+22*(len(rows)-1),.75,INK)


def main(root):
    metrics=list(csv.DictReader((root/'metrics.csv').open(encoding='utf8')))
    ex=json.loads((root/'convex-distance.json').read_bytes())['runs']
    swaps=json.loads((root/'pose-swap.json').read_bytes())['pairs']
    out=root/'review'
    c=Canvas(str(out/'review.pdf'),pagesize=(W,H),pageCompression=1)
    c.setTitle('单门穿越中的臂形、时序与净空 | G1 组会审阅')
    header(c,1,'同一扇门下，弯臂给夹爪留出多少空间？','12 趟窄门对照和 1 趟宽门链路检查。门后夹爪目标、截止时间和飞控参数对四组相同。')
    text(c,36,138,'最近点出现在门楣和门槛的不同位置',14,'B')
    image(c,out/'closest.png',34,145,562,240)
    para(c,401,['图示种子 241119。红线连接三维碰撞几何的最近点，各帧取本组最小时刻。',
                '门楣与门槛按中央剖面绘出。评分使用两侧门柱、完整门框和全部 372 个注册部件与代理几何。'])
    text(c,36,457,'逐组结果',14,'B')
    rows=[]
    for k,title in [('straight','直臂'),('fixed','固定 40°'),('variable_nominal','可变形 · 常规'),('variable_timed','可变形 · 调时')]:
        v=[r for r in metrics if r['mode']==k]
        med=lambda part:statistics.median([100*ex[r['case']][part]['upper_m'] for r in v])
        rows.append([title,str(sum(r['pass_all']=='True' for r in v))+'/3','%.2f'%med('body'),'%.2f'%med('arm_tool'),
                     '%.2f'%statistics.median(float(r['final_error_cm']) for r in v)])
    table(c,474,['形态与时序','协议通过','机身净空 (cm)','臂与爪净空 (cm)','截止时末端误差 (cm)'],rows,[106,65,113,119,155])
    para(c,617,['净空列为每趟全程凸体表面最小距离的组内中位数，三维距离求解区间最宽 0.14 mm。',
                '通过列依照事先冻结的包围盒下界评分，机身须达 10 cm，臂和工具须达 7 cm。',
                '直臂三趟均过门到位且无碰撞。凸体复核有两趟低于 10 cm，一趟略高于 10 cm。',
                '固定 40° 也三趟通过。在这扇门上，可变形方案的收益体现在额外净空。'])
    line(c,36,700,594,700)
    para(c,724,['到位要求为整机越门后，夹爪距目标 ≤ 8 cm、轴向误差 ≤ 25°，连续保持 1 s。',
                '所有窄门工况均在约 43 s 达成到位判据，截止时间同为 52 s。'])
    c.showPage()
    header(c,2,'相同弯角为何产生了不同的过门净空？','可变形两组共用机身位置参考和 75° 过门指令。调时组提前弯臂、延后收臂，且两段动作均放慢。')
    image(c,out/'curves.png',35,117,560,320)
    para(c,450,['a–c 为预选种子 241119，曲线显示完整部件包围盒下界和实际弯角。d 的每条线为同一噪声',
                '种子的两趟独立飞行，使用凸体表面距离。三组臂与爪最小净空分别增加 3.99、4.14、3.91 cm。'])
    text(c,36,510,'在同一个飞行时刻交换姿态记录',14,'B')
    table(c,524,['噪声种子','常规整机','常规机身＋调时臂','调时机身＋常规臂','调时整机'],[
        [str(r['seed'])]+['%.2f'%r[k] for k in ('nominal_actual_cm','nominal_base_timed_arm_cm','timed_base_nominal_arm_cm','timed_actual_cm')] for r in swaps],[80,91,144,144,99])
    para(c,638,['单位为 cm。中间两列是离线几何重放。在种子 241119 的最近帧，调时机身高约 4.2 cm。',
                '三对记录中，交换臂形仅带来毫米以下的数值变化；交换机身姿态与位置后重现约 4 cm 差异。'.replace('；','，')])
    line(c,36,670,594,670)
    text(c,36,695,'验证范围与论文去向',12.3,'B')
    para(c,718,['PX4 EKF2 驱动四桨，MuJoCo 积分机身和臂。ULog、ROS bag 与实际施加的推力已对时。',
                '窄门采用有界单门候选搜索。多窗口接管、相机建图、主动反力矩补偿及实体标定仍待验证。',
                '本页供组会审阅。多门滚动结果与失效边界确定后，再选入论文主图。'])
    c.showPage();c.save()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',type=Path);main(ap.parse_args().stage)
