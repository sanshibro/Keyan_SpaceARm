"""Two-page Chinese meeting proof. Generate after coupled replay and independent audit."""
import argparse,json,io
from pathlib import Path
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from PIL import Image
PAGES=(630,816);INK='#20383e';MUTED='#52636b';TEAL='#286f63'
def read(p):return json.loads(p.read_bytes())
def main(root,fonts):
    for alias,source in [('body','msyh.ttc'),('bold','msyhbd.ttc')]:pdfmetrics.registerFont(TTFont(alias,str(fonts/source)))
    s=read(root/'markerless-scores.json');bench=read(root/'bench-scores.json');g=read(root/'geometry.json')
    audit=read(root/'audit.json');assert len(audit['flight_verification'])==4
    rd=root/'review';c=Canvas(str(rd/'review.pdf'),pagesize=PAGES,pageCompression=1)
    c.setTitle('视觉失效时，柔性臂如何维持形态估计？')
    c.setAuthor('空中连续体操作研究组')
    def text(x,y,value,sz=10,bold=False,color=INK,right=False):
        c.setFillColor(HexColor(color));c.setFont('bold' if bold else 'body',sz)
        (c.drawRightString if right else c.drawString)(x,816-y,value)
    def line(y,w=.65):
        c.setStrokeColor(HexColor('#b9c6c5'));c.setLineWidth(w);c.line(35,816-y,595,816-y)
    def para(y,words,sz=9.75,dy=15):
        for i,v in enumerate(words):
            assert pdfmetrics.stringWidth(v,'body',sz)<564,v
            text(35,y+i*dy,v,sz)
    def header(n,subject,subtitle):
        text(35,26,'SUPER  /  QINGYU    空中连续体操作',8.5,color=MUTED)
        text(595,26,'单门仿真重放    /    本体感知    /    2026.11.19',8.5,color=MUTED,right=True)
        line(38,1);text(35,76,subject,21,True);text(35,99,subtitle,9.3,color=MUTED)
        line(780);text(35,800,'组会审阅   ·   PX4 / MuJoCo 已记录轨迹   ·   独立虚拟动捕评分',8.1,color=MUTED)
        text(595,800,'%02d  /  02'%n,8.5,color=MUTED,right=True)
    def image(name,y,w=560):
        im=Image.open(rd/name).convert('RGB');h=w*im.height/im.width
        memory=io.BytesIO();im.save(memory,format='JPEG',quality=92,subsampling=0);memory.seek(0)
        c.drawImage(ImageReader(memory),35,816-y-h,w,h);return y+h
    def table(y):
        labels=['弯曲量 RMSE','视觉正常全程','遮挡 4 s 内','遮挡期间覆盖率']
        widths=[171,128,125,136];line(y,.9)
        for r,vals in enumerate([labels]+rows):
            x=35
            for j,(v,width) in enumerate(zip(vals,widths)):
                if j==0:text(x+4,y+16+20*r,v,9.2,r==0)
                else:text(x+width-7,y+16+20*r,v,9.2,r==0,right=True)
                x+=width
            if r==0:line(y+22)
        line(y+23+20*len(rows),.8)
    header(1,'相机失效时，柔性臂怎样继续估计弯曲量？','三种传感方案共享运动轨迹、相机外参和时钟，并额外控制滤波本身的影响。')
    image('mechanism.png',112)
    text(35,408,'先用独立真值学应变，再检验视觉',13,True,TEAL)
    para(432,['在臂体四周每隔 90° 取一条 60 mm 表面路径，以相对伸长率作为等效应变。经四路增益、漂移、',
              '45 ms 滞后、噪声和 12 位量化形成 50 Hz ADC。安装座与末端的虚拟动捕点给出训练角度。',
              '按完整动作拆为 8 段训练、2 段验证、3 段测试。测试弯曲量 RMSE 为 %.2f°，平面角为 %.2f°。'%(
               bench['test']['theta_rmse_deg'],bench['test']['phi_rmse_deg'])])
    text(35,496,'从腕部图像到弯臂角度',13,True,TEAL)
    para(520,['用 NUC 上的 ORB-SLAM3 RGB-D 处理 480 × 360 的腕部彩色图像和公制度量深度。每秒 25 帧，',
              '场景纹理固定在环境中。中立姿态、安装外参与 PX4 机身姿态将相机姿态换算到臂根坐标系。',
              '仅接收原始地图中跟踪有效的位姿。已知同一时刻的动捕真值只在评分时进入飞行比较。'])
    text(35,584,'融合和失效工况怎样实现',13,True,TEAL)
    para(608,['按本机专利第 5 章的思路，应变验证误差决定应变方差，SLAM 跟踪点数和重投影残差决定视觉',
              '权重。弯曲平面角在单位圆上融合，近伸直时标记方向不可辨。视觉延迟统一设为 0.20 s，',
              '滤波器在曝光时刻更新并用期间的应变读数推到当前时刻。训练参数在飞行测试中冻结。',
              '同一趟飞行分别重放正常画面与遮挡画面。第 31 至 35 s 将 RGB 和深度同时置零，重新运行',
              '原生跟踪器。对应输出时间为第 31.2 至 35.2 s。飞行比较共四趟，包含两个种子和两种时序。'])
    para(712,['平面角仅在真值弯角至少 5° 时评分。虚拟贴片长度、滞后与噪声均为仿真参数，实机需单独标定。',
              '图中示例为 241119 的状态触发策略。上排为同尺度侧视剖面，下排为实际送入跟踪器的画面。'],8.7)
    c.showPage()
    header(2,'融合保住了哪些观测？','先看输出是否连续，再在相同时间点比较误差。仅视觉失效时，角度误差无定义。')
    image('curves.png',107,553)
    n=s['conditions']['natural']['all'];d=s['conditions']['occluded']['during']
    rows=[]
    for m,label in [('strain','仅应变 · 原始预测'),('strain_filter','仅应变 · 相同滤波器'),('slam','仅 ORB-SLAM3'),('fusion','应变 + ORB-SLAM3')]:
        val=d[m]['all']['theta_rmse_deg']
        rows.append([label,'%.3f°'%n[m]['all']['theta_rmse_deg'],'无可用位姿' if val is None else '%.3f°'%val,
                     '%.0f%%'%(d[m]['all']['coverage']*100)])
    table(438)
    text(35,572,'四次揭开遮挡，四次都建立了新地图',12.8,True,TEAL)
    para(594,['四次恢复后都获得了局部视觉跟踪。到第 46 s，新地图与初始地图的相对方向仍未确定，',
              '因此融合保持应变更新。遮挡组全程，视觉原地图位姿覆盖率为 %.2f%%。正常段中融合与应变'%(
                  100*s['conditions']['occluded']['all']['slam']['all']['coverage']),
              '平均精度很接近。本轮直接支持视觉失效后的输出连续性；两者正常段的精度差异很小。'.replace('；','，')])
    geom=g['conditions']['natural']['fusion']
    text(35,657,'与避障的衔接',12.8,True,TEAL)
    para(679,['同一估计臂形重建完整几何。四趟记录、每趟 80 个时刻和 33 种假设门槛高度共产生 10,560 个',
              '相关判定。加入此前两趟校准得到的 %.2f cm 余量后，逐帧错误放行由 %d 次变成 %d 次。'%(
                g['prior_empirical_pad_m']*100,geom['frame_raw']['false_pass'],geom['frame_with_prior_pad']['false_pass']),
              '整条路线安全候选获准数由 %d 减至 %d（参考安全候选 %d 个）。下一步先检验原地图重定位，'%(
                geom['route_raw']['safe_released'],geom['route_with_prior_pad']['safe_released'],geom['route_raw']['reference_pass']),
              '再在同一单门任务中以估计臂形和不确定度闭环放行，比较实际最小净空、停等与任务完成率。'],9.2)
    c.showPage();c.save()
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',type=Path);a.add_argument('--fonts',type=Path,default=Path('C:/Windows/Fonts'))
    args=a.parse_args();main(args.stage,args.fonts)
