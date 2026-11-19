"""Build the two-page Chinese group review from the scored replay and its source frames."""
import argparse,io,json
from pathlib import Path
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

def build(root):
    rd=root/'review'; get=lambda n:json.loads((root/n).read_bytes())
    scores=get('paired-scores.json'); ab=get('support-scores.json'); check=get('audit.json');bench=get('variance-bench.json')
    for k,fn in [('C','msyh.ttc'),('B','msyhbd.ttc')]:
        pdfmetrics.registerFont(TTFont(k,str(Path('C:/Windows/Fonts')/fn)))
    W,H=595,792
    c=Canvas(str(rd/'review.pdf'),pagesize=(W,H),pageCompression=1)
    c.setTitle('腕部视觉何时能纠正应变偏差？四趟同轨重放')
    ink='#172e37';grey='#53666b';green='#196656'
    def label(x,y,s,size=9.3,f='C',color=ink,right=False):
        c.setFillColor(HexColor(color));c.setFont(f,size)
        (c.drawRightString if right else c.drawString)(x,H-y,s)
    def line(y,weight=.55,color='#c0cdcc'):
        c.setStrokeColor(HexColor(color));c.setLineWidth(weight);c.line(35,H-y,560,H-y)
    def lines(y,arr,size=9.25,leading=15):
        for i,st in enumerate(arr):
            assert pdfmetrics.stringWidth(st,'C',size)<=525,st
            label(35,y+i*leading,st,size)
    def title(n,s,deck):
        label(35,26,'空中连续体操作研究组    /    SUPER × Qingyu',8.2,color=grey)
        label(560,26,'G1 单门 · 腕部视觉与应变融合',8.2,color=grey,right=True);line(37,.9)
        label(35,73,s,19.7,'B');label(35,94,deck,9.1,color=grey)
        line(752);label(35,770,'组会审阅    /    四趟既有飞行同轨重放    /    2026.11.19',8,color=grey)
        label(560,770,'%d / 2'%n,8.2,color=grey,right=True)
    def subsection(y,s):label(35,y,s,12.2,'B',green)
    def raster(p,x,y,w):
        with Image.open(p) as src:
            im=src.convert('RGB');im.thumbnail((1280,900));buf=io.BytesIO();im.save(buf,'JPEG',quality=82,optimize=True,subsampling=0)
            buf.seek(0);hh=w*im.height/im.width;c.drawImage(ImageReader(buf),x,H-y-hh,w,hh)
    def table(y,head,rows,widths):
        line(y,.9)
        for i,row in enumerate([head]+rows):
            x=35;yy=y+15+19*i
            for j,(v,w) in enumerate(zip(row,widths)):
                right=j>0;f='B' if i==0 else 'C'
                assert pdfmetrics.stringWidth(v,f,8.9)<w-4,v
                label(x+w-4 if right else x+2,yy,v,8.9,f,right=right);x+=w
            if i==0:line(y+20)
        line(y+19*(len(rows)+1)+3)
    title(1,'腕部视觉何时能纠正应变偏差？','在同一架仿真飞机、同一组动作和同一采样时钟下，只改变腕部图像与应变观测。')
    raster(rd/'tracking.png',34,110,527)
    lines(441,['图 1　上排是实际送入 RGB-D ORB-SLAM3 的腕部图像。曲线按估计器输出时间绘制，',
         '曝光至可用延迟为 0.20 s。浅色带为逐步遮挡，深色带为完全遮挡。下图为存在零点偏移的一趟。'],8.75,13.7)
    subsection(493,'两路观测各自回答“这一帧有多准”')
    lines(517,['视觉先检查跟踪状态与地图坐标，再按匹配数量、匹配比例、重投影误差与连续跟踪时长估计',
           '置信度。应变用四路采集值、相对两片的和与差、读数下限、变化速率和近期运动估计方差。',
           '弯角按逆方差加权，弯曲方向按圆周加权。靠近伸直时，方向可观测性随弯角降低。'])
    subsection(590,'如何让视觉教应变，同时检验学到的修正')
    lines(613,['应变角度模型用虚拟动捕训练。每次留出一整段台架动作，取得训练方差模型所需的残差。共',
           '8 段训练、2 段标定、3 段测试。视觉方差在另一次飞行的 800 帧上标定。在线仅更新弯角偏差。',
           '连续高质量的视觉观测形成候选修正，后续 6 个独立于提案的合格观测用于验收，失视时冻结。'])
    lines(690,['组会问题　局部遮挡时，剩下的窄条仍能给出大量低残差匹配。图 1a 的红线额外考虑特征的',
           '二维分布。该检查是在主试验后增加的探索性对照；图 1b 保留原先登记的三种融合方法。'],9)
    c.showPage()
    title(2,'应变健康与带偏时，结论不同','弯曲量对独立虚拟动捕的 RMSE，单位为度。各行使用同一批四趟轨迹、同一段时间。')
    def row(st,op,ph,lab):
        a=scores[st][op][ph]
        def cell(m):
            v=a[m]['all']['theta_rmse_deg'];return '无有效帧' if v is None else f'{v:.2f}'
        return [lab]+[cell(m) for m in ['strain','vision','previous','adaptive','corrected']]
    rows=[]
    for op,ph,lab in [('natural','all','清晰'),('occluded','same_four_seconds','全遮挡')]:
        for st,sn in [('nominal','正常'),('zero_offset','零点偏移'),('material_memory','迟滞与回落')]:
            rows.append(row(st,op,ph,lab+' · '+sn))
    table(118,['视觉和应变工况','仅应变','仅视觉','原融合','自适应','加偏差校正'],rows,[153,74,73,72,72,81])
    lines(273,['表 1　清晰段取 14–46 s，共 6,400 个输出时刻。全遮挡取 31.20–35.20 s，共 800 个。',
               '“仅应变”沿用新版时序滤波；“仅视觉”按到达时刻评分。三种融合使用相同的视觉姿态输入。'],8.75)
    subsection(327,'为什么权重不能只看画面是否清晰')
    lines(350,['视觉清晰且应变有零点偏移时，在两路同时有新测量的时刻，视觉对弯角观测的平均权重从',
           '4.1% 升至 87.1%。应变正常时新版仍给视觉 84.2% 的权重，牺牲了原有精度。留出动作上，',
           '正常应变的实测弯角 RMSE 为 %.2f°，预测标准差均方根却有 %.2f°；混合扰动的 95%% 名义区间'%(
               bench['nominal']['theta']['rmse_deg'],bench['nominal']['theta']['rms_sigma_deg']),
           '只覆盖 %.1f%% 的误差。几种应变工况共用的方差模型尚未校准充分。'%(100*bench['mixed']['theta']['in_95_fraction'])])
    subsection(433,'局部遮挡、短时丢失与重建地图')
    lines(456,['在 88%% 宽度被遮住的区间，考虑特征分布后，正常应变的融合误差从 %.2f° 降至 %.2f°；'%(
                scores['nominal']['graded']['weak']['adaptive']['all']['theta_rmse_deg'],ab['graded/nominal']['partial']['theta_rmse_deg']),
           '有应变零偏时则从 %.2f° 升至 %.2f°。这是同一组图像上的探索性结果，需要新轨迹验证。'%(
                scores['zero_offset']['graded']['weak']['adaptive']['all']['theta_rmse_deg'],ab['graded/zero_offset']['partial']['theta_rmse_deg']),
           '短遮挡 0.32 s 后，四趟均由原生跟踪恢复，解除遮挡至输出用时 0.20–0.28 s。长遮挡 4 s 后，',
           '原生跟踪进入新地图。通过 RGB-D 关键帧几何配准恢复原坐标，用时 0.48–0.49 s（含传输）。'])
    subsection(541,'可以据此写下的结论')
    lines(564,['视觉可抑制应变零偏，应变可在失视时维持连续输出。这两项作用须分别检验。在 4 s 全遮挡段，',
           '三种应变工况各有 800/800 个有效融合输出，在线修正次数均为零。迟滞与回落工况从 3.69°',
           '改善至 2.04°；带偏应变仍有 8.17° 误差。下一轮先辨识应变零点与回程状态，再用全新轨迹',
           '检验正常、偏移和两路同时退化。当前数据衡量估计精度，飞行控制效果仍待闭环试验。'])
    line(648)
    lines(670,['方法依据　组内专利设计第 4.2、5.1–5.3 节；Campos 等，IEEE T-RO 37 (2021)，1874–1890；',
            'Kendall 与 Gal，NeurIPS 30 (2017)。四次估计重放沿用既有 PX4 / MuJoCo 轨迹。'],8.6)
    c.linkURL('https://doi.org/10.1109/TRO.2021.3075644',(218,H-674,514,H-658))
    c.linkURL('https://proceedings.neurips.cc/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html',(35,H-689,225,H-674))
    c.showPage();c.save()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',type=Path);build(parser.parse_args().stage.resolve())
