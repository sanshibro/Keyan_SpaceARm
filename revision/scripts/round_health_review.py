"""Render the two-page review using the frozen test results, captured images and shared flight clock."""
import io, json, sys
from pathlib import Path
from PIL import Image
from reportlab import rl_config
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

def build(root):
    read=lambda n:json.loads((root/n).read_bytes())
    s=read('flight-scores.json');now=read('timing-reports.json')['flight']
    audit=read('audit.json');freeze=read('freeze.json');rd=root/'review'
    for code,file in [('R','msyh.ttc'),('B','msyhbd.ttc')]:
        pdfmetrics.registerFont(TTFont(code,str(Path('C:/Windows/Fonts')/file)))
    rl_config.useA85=0
    W,H=595,792;L=36;R=559;ink='#182f36';muted='#476068';accent='#196b5b';rust='#904a34'
    c=Canvas(str(rd/'review.pdf'),pagesize=(W,H),pageCompression=1)
    c.setTitle('四路应变能区分零点偏移和材料迟滞吗？');c.setAuthor('空中连续体操作研究组')
    def text(x,top,value,sz=9.9,font='R',color=ink,align='left'):
        c.setFillColor(HexColor(color));c.setFont(font,sz)
        {'left':c.drawString,'right':c.drawRightString,'center':c.drawCentredString}[align](x,H-top,value)
    def rule(top,width=.6,color='#b4c5c6',xl=L,xr=R):
        c.setStrokeColor(HexColor(color));c.setLineWidth(width);c.line(xl,H-top,xr,H-top)
    def para(top,lines,sz=10,leading=16.8,color=ink):
        for i,line in enumerate(lines):
            assert pdfmetrics.stringWidth(line,'R',sz)<R-L,line
            text(L,top+i*leading,line,sz,color=color)
    def head(page,title,sub):
        text(L,29,'空中连续体操作研究组    /    G1 单门实验',8.2,color=muted)
        text(R,29,'SUPER × Qingyu    /    形态感知',8.2,color=muted,align='right');rule(40,1)
        text(L,75,title,20.4,'B');text(L,97,sub,9.25,color=muted)
        rule(751);text(L,770,'组会审阅    /    三趟新飞行仿真    /    2026.11.20',8,color=muted)
        text(R,770,str(page)+' / 2',8,color=muted,align='right')
    def sec(top,title):text(L,top,title,12.2,'B',accent)
    def figure(p,x,top,w):
        with Image.open(p) as orig:
            im=orig.convert('RGB');im.thumbnail((1400,820));buf=io.BytesIO()
            im.save(buf,'JPEG',quality=82,optimize=True,subsampling=0);buf.seek(0)
            height=w*im.height/im.width;c.drawImage(ImageReader(buf),x,H-top-height,w,height)
    head(1,'四路应变能区分零点偏移和材料迟滞吗？','用虚拟动捕训练应变模型，再用三趟独立飞行检验它在腕部相机被遮挡时能否估准臂形。')
    figure(rd/'mechanism.png',L,116,R-L)
    para(392,['图 1　左图为同一趟左偏航迹上的机体与腕部图像。右图灰带为融合器收到黑帧的 4 秒。',
              '曲线 c 为估计角减去虚拟动捕角，绿色带为名义 95% 区间。绿带越过零线才覆盖该时刻真值。'],8.7,14,muted)
    sec(451,'应变采集、真值和视觉各自从哪里来')
    para(477,['四片 Slim M 沿臂中段轴向布置，原长 25 mm，预拉伸约 30%，直臂安装长度为 32.5 mm。',
              '从 MuJoCo 臂表面的长度变化生成四路读数，沿用柔致采集盒的实测采样间隔与噪声特征。',
              '采集盒的三路历史信号提供偏移量级，第四路量级为估计。材料加载、卸载和回落按设定范围模拟。',
              '独立虚拟动捕追踪安装座和工具的标记。腕部 RGB-D 图像送入原生 ORB-SLAM3，按时间戳融合。'],9.5,16.2)
    sec(570,'用从未参加训练的动作来回答问题')
    para(595,['14 段台架动作按整段划分，8 段训练、3 段标定、3 段测试。动作包含反向、不同速度和停驻。',
              '相对贴片的读数差提供弯曲信息。两片平均值偏离正常关系的程度，以及近期的读数变化，',
              '用于估计偏移和回程状态。如果若干状态都能解释当前读数，估计方差也计入它们的角度分歧。',
              '冻结参数后，用三趟新 PX4 SITL / MuJoCo 轨迹复核。飞行日志包含 ULog、ROS bag 和物理状态。',
              '本轮在飞行记录上检验观测器，飞控执行记录中的指令。'],9.5,16.2)
    para(708,['三趟轨迹弯曲峰值为 65.3°、76.4°、82.3°。每趟同一组 850 帧分别以清晰和全遮挡条件送入视觉。',
              '明暗条件共处理 5,100 帧次。每趟 4 秒为局部试验窗口，整趟飞行其余时段仍有视觉。'],8.55,14,muted)
    c.showPage()
    head(2,'能报出偏移，仍可能估错弯角','弯曲量相对虚拟动捕的 RMSE，单位为度。每种应变工况使用同一组飞行、同一视觉和评分时钟。')
    xs=[L,163,223,282,338,395,453,R];cent=[(xs[i]+xs[i+1])/2 for i in range(7)]
    rule(120,.95);text(L+2,137,'应变工况',9,'B')
    text((xs[1]+xs[3])/2,137,'视觉清晰 · 32 s',9.2,'B',align='center')
    text((xs[3]+xs[6])/2,137,'完全失视 · 4 s',9.2,'B',align='center')
    text(cent[6],137,'失视区间覆盖',8.8,'B',align='center')
    for i,h in enumerate(['','上轮','本轮','固定噪声','上轮','本轮','本轮名义 95%']):
        if h:text(cent[i],155,h,8.1,color=muted,align='center')
    rule(163)
    for i,(state,label) in enumerate([('healthy','健康'),('offset','零点偏移'),('memory','材料回程'),('mixed','偏移与回程叠加'),('differential','相对贴片反向偏移')]):
        cl=s[state]['natural']['all'];bl=s[state]['occluded']['blind_window'];y=181+i*22
        text(L+2,y,label,9.25)
        vals=[p[k]['all']['theta_rmse_deg'] for p,k in [(cl,'previous_teacher'),(cl,'health'),(bl,'legacy'),(bl,'previous_teacher'),(bl,'health')]]
        for col,val in enumerate(vals,1):
            text(cent[col],y,format(val,'.2f'),9.2,'B' if col in (2,5) else 'R',accent if col in (2,5) else ink,'center')
        cov=bl['intervals']['reported_sigma']['coverage']*100
        text(cent[6],y,format(cov,'.1f')+'%',9.2,color=rust if cov<90 else ink,align='center')
    rule(280,.8)
    para(300,['表 1　清晰段每行 4,800 个时刻，失视段每行 600 个时刻。三趟均从释放后 31.2 s 起评分 4 秒。',
              '“上轮”保留视觉在线校偏，“本轮”使用应变健康估计且关闭在线校偏。“固定噪声”为原基线。',
              '各行汇总同轨误差。覆盖率为实际弯角落入所报区间的时刻占比。'],8.65,14,muted)
    sec(371,'漂移有多少能检出，迟滞又有多少能分清')
    ma=now['macro_balanced_accuracy'];alarms=now['sustained_alarm']
    para(395,[f'逐趟等权计算平衡准确率，常规注入零偏为 {ma["offset"]*100:.1f}%，材料迟滞为 {ma["memory"]*100:.1f}%。',
              f'在 6 组含常规偏移的飞行重放中，持续报警延迟为 {alarms["min_delay_s"]:.2f}–{alarms["max_delay_s"]:.2f} s。四种工况合计的偏移前及无偏移段',
              '观察到 0 次持续误报。该报警要求零偏概率连续五个采样点达到 90%。这里只覆盖设定的偏移类型。',
              '健康应变配清晰视觉时，原固定噪声基线为 %.2f°，本轮为 %.2f°。当前融合付出了健康精度代价。'%(
                    s['healthy']['natural']['all']['legacy']['all']['theta_rmse_deg'],s['healthy']['natural']['all']['health']['all']['theta_rmse_deg'])],9.25,16)
    sec(490,'为什么四片传感器有时分不清实际弯了多少')
    para(514,['构造两段最终分别为 35° 和 43° 的动作。给其中一段的第 1、3 路加入最大 29 和 26 个计数的',
              '定向零偏，其余两路保持原样，两段动作便能产生完全相同的四路读数历史。此时仅靠应变输出',
              '同一个角度，至少一个结果会错 4°。这个反例说明可观测性的边界，出现概率需要实物数据估计。',
              '遮挡期每种应变工况均有 600/600 个融合输出。偏移与回程叠加时，名义 95% 区间实际只覆盖',
              '56/600 个时刻，定向偏移时为 0/600。连续给出读数和有足够精度是两个独立的检验。'],9.3,16)
    sec(628,'下一项实验')
    para(652,['用已知直臂姿态或持续高质量视觉锚定缓慢变化的零点，再在长遮挡中检验区间能否随时间扩大。',
              '同步采四路原始计数和独立角度真值，做同角度加载、卸载和停驻。分别检验健康精度、失视',
              '误差与覆盖率，结果决定估计器何时可以接入规划器。'],9.5,16)
    para(718,['冻结模型 SHA-256 前缀 '+freeze['model_sha256'][:16]+'    /    保留逐趟评分、原始遥测与输入时序校验。'],8.25,14,muted)
    c.showPage();c.save()
if __name__=='__main__':build(Path(sys.argv[1]).resolve())
