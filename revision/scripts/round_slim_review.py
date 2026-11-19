"""Create the two-page meeting proof with native vector plots and selectable text."""
import argparse,io,json
from pathlib import Path
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader,PdfWriter,Transformation
def read(p):return json.loads(p.read_bytes())
def build(stage):
    rd=stage/'review';metrics=read(stage/'memory-scores.json');flight=read(stage/'flight-scores.json')['primary']['conditions']
    audit=read(stage/'audit.json');sw=read(stage/'sweep.json')
    fonts=Path('C:/Windows/Fonts')
    for k,f in [('C','msyh.ttc'),('B','msyhbd.ttc')]:pdfmetrics.registerFont(TTFont(k,str(fonts/f)))
    file=io.BytesIO();W,H=630,816;c=Canvas(file,pagesize=(W,H),pageCompression=1);placements=[[],[]]
    ink='#213538';muted='#536568';green='#24675b'
    def text(x,y,words,size=10,font='C',color=ink,align='left'):
        c.setFont(font,size);c.setFillColor(HexColor(color))
        (c.drawString if align=='left' else c.drawRightString)(x,H-y,words)
    def rule(y,x=34,x2=596,color='#becac7',lw=.65):
        c.setStrokeColor(HexColor(color));c.setLineWidth(lw);c.line(x,H-y,x2,H-y)
    def para(y,lines,size=9.7,leading=16):
        for i,line in enumerate(lines):
            assert pdfmetrics.stringWidth(line,'C',size)<566,line
            text(34,y+leading*i,line,size)
    def head(n,heading,sub):
        text(34,28,'空中连续体操作研究组   /   SUPER · Qingyu',8.6,color=muted);text(596,28,'G1 单门    /    应变本体感知    /    2026.11.19',8.6,color=muted,align='right')
        rule(39,lw=.95);text(34,77,heading,21,'B');text(34,101,sub,9.3,color=muted)
        rule(779);text(34,800,'组会审阅   ·   设备资料、既有采集盒记录和仿真同轨重放',8.2,color=muted);text(596,800,'%02d / 02'%n,8.5,color=muted,align='right')
    def sub(y,s):text(34,y,s,13,'B',green)
    def table(y,head,rows,widths):
        rule(y,lw=.95);cells=[head]+rows
        for i,row in enumerate(cells):
            x=34
            for j,(v,w) in enumerate(zip(row,widths)):
                assert pdfmetrics.stringWidth(v,'B' if not i else 'C',9.1)<w-3,v
                text(x+3 if not j else x+w-4,y+15+i*20,v,9.1,'B' if not i else 'C',align='left' if not j else 'right');x+=w
            if i==0:rule(y+20)
        rule(y+len(cells)*20+2)
    head(1,'弯曲时，四条贴片实际测到了什么？','由贴片表面长度算应变，再由产品曲线和采集盒读数估计臂形。')
    placements[0].append(('mechanism.pdf',29,120,568))
    sub(337,'25 mm 是敏感段自然长度，32.5 mm 是安装跨度')
    para(362,['实物为 Slim M。主模型将四条 25 × 3.5 mm 的敏感段贴在 280 mm 臂体的中点，沿周向间隔 90°。',
              '30% 预拉伸得到 32.5 mm 的中立跨度。每条长度来自 MuJoCo 臂节运动后的表面路径。按自然',
              '长度计算拉伸率，沿厂商 25 °C 曲线插值得到相对电阻。采集盒按旧串口记录约 40.2 Hz 取样。'])
    rows=[]
    for key,l in [('p00','25 mm · 自然贴合'),('p15','28.75 mm · 预拉 15%'),('p30','32.5 mm · 预拉 30%'),('p40','35 mm · 预拉 40%')]:
        item=sw[key];zero=audit['adc_sweep'][key]['first_nominal_low_code_bend_deg']
        rows.append([l,'%.1f%% / %.1f%%'%(item['min_120_percent'],item['max_120_percent']),
               '全程保持伸长' if item['first_slack_test_angle_deg'] is None else '从 %d° 松弛'%item['first_slack_test_angle_deg'],
               '未触及' if zero is None else '%d°'%zero])
    table(424,['中立安装','120° 时内 / 外侧','内侧贴合','首次触及读数下限'],rows,[170,133,136,123])
    para(548,['在 24 个弯曲平面按 5° 扫角。负值表示模型中的表面路径短于贴片自然长度。采集下限使用历史',
              '记录估计的通道增益与零点，未加入随机噪声。实际是否提前触底取决于本机的档位和调零。'],9.2)
    sub(606,'一次过门里运动顺畅，仍可能有跟踪误差')
    para(631,['四趟已有 PX4 / MuJoCo 飞行的 14–46 s 内，参考动捕最大弯角约 72.7°。32.5 mm 安装的',
              '5,149 次采集均在拉伸及有效读数范围内。各趟下发角对虚拟动捕的跟踪 RMSE 为 1.83°–1.88°。',
              '中段模拟应变用 8 段台架动作的独立动捕训练，2 段验证，3 段测试。主模型留出测试弯角',
              'RMSE 为 0.28°。这个数衡量估计误差。采集盒旧资料只有三个有效通道，第四路的长度独立',
              '计算，增益取三路中位数，噪声取独立抽样的静止记录。'])
    text(34,740,'资料依据  北京柔致《Elastreme Sense Slim 技术文档》第 1–3 页及柔致采集盒旧串口记录。',8.5,color=muted)
    c.showPage()
    head(2,'同样的长度，拉上去和退回来读数一样吗？','把往返迟滞、定长保持时的回落和机械臂的跟踪误差分开检验。')
    placements[1].append(('memory.pdf',29,116,568))
    sub(316,'产品资料限定到哪里')
    para(339,['手册给出 0–100% 拉伸量程、约 0.01% 分辨率及小于 0.1 ms 的响应时间。第 2 页展示了 10%、',
              '20%、50%、80% 的反复拉伸。50% 拉伸循环一万次后的电阻变化小于 1%，描述的是耐久性。',
              '该页没有同步长度与卸载支路。示例在相对电阻中另设 0.02 往返支路间距及 2 s 衰减时间，',
              '用于检验估计器。材料的即时输出仍保留。旧采集盒另拟合了一组安装后的等效动态参数。'],9.45)
    sub(420,'同四趟飞行、同一相机遮挡、同一动捕真值')
    sim=metrics['simulated']
    rows=[]
    for lab,tag in [('静态换算 · 原预测器',None),('往返迟滞 · 原预测器','loop'),('仅保持回落 · 原预测器','relax'),('迟滞与回落 · 原预测器','combined'),('迟滞与回落 · 重训预测器','combined_retrained')]:
        n=flight['natural']['all']['fusion']['all']['theta_rmse_deg'] if tag is None else sim[tag]['natural']['fusion']['theta_rmse_deg']
        b=flight['occluded']['during']['fusion']['all']['theta_rmse_deg'] if tag is None else sim[tag]['occluded']['fusion']['theta_rmse_deg']
        rows.append([lab,'%.2f°'%n,'%.2f°'%b])
    table(434,['融合弯角 RMSE','视觉正常 14–46 s','视觉遮挡 4 s'],rows,[235,167,160])
    para(576,['每次遮住腕部 RGB-D 相机 4 s，原地图视觉均无可用角度，四趟融合共覆盖 800 / 800 个输出时刻。',
              '有输出仍可能有偏差。表中迟滞大小为灵敏度试验设定。重训沿用相同八段训练动作及静态特征。'],9.2)
    sub(629,'实物数据对照，以及实机还缺哪一步')
    para(654,['在一段旧实物记录中估计等效动态参数，在另一段独立记录中检验。第一路预测误差由 39.8 降到',
              '24.5 个计数，另两路由 42.0 降到 40.9、由 63.0 降到 62.5。历史相机测的是整臂角度，',
              '黏接、传动和时间对齐均可能贡献差异。建议固定弯曲平面与灵敏度，同步记录四路原始码和',
              '贴片两端标记间的表面长度，分别做慢速、快速往返与定长保持，再按完整循环留出检验。'],9.4)
    text(34,745,'对照方法  Di Tocco 等，Sensors 22 (2022) 3954，图 5、7、8。',8.5,color=muted)
    c.linkURL('https://doi.org/10.3390/s22103954',(34,H-748,344,H-734),relative=0)
    c.showPage();c.save();file.seek(0);doc=PdfReader(file);writer=PdfWriter()
    for i,page in enumerate(doc.pages):
        for name,x,top,width in placements[i]:
            f=PdfReader(rd/name).pages[0];scale=width/float(f.mediabox.width)
            page.merge_transformed_page(f,Transformation().scale(scale).translate(x,H-top-float(f.mediabox.height)*scale),over=True)
        added=writer.add_page(page);added.compress_content_streams()
    writer.compress_identical_objects(remove_identicals=True,remove_orphans=True)
    writer.add_metadata({'/Title':'Slim M 中段应变、采集盒与往返迟滞','/Author':'空中连续体操作研究组'})
    with (rd/'review.pdf').open('wb') as f:writer.write(f)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',type=Path);build(ap.parse_args().stage.resolve())
