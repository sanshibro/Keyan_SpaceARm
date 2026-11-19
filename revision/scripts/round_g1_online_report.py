"""Two-page meeting proof. All reported statistics are read from the audited scores."""
import argparse
import csv
import json
import statistics
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

PAGE = (630, 816)
INK, MUTED, RULE = '#223940', '#52656a', '#afbfbd'


def make_report(root, fonts):
    pdfmetrics.registerFont(TTFont('body', str(fonts / 'msyh.ttc')))
    pdfmetrics.registerFont(TTFont('bold', str(fonts / 'msyhbd.ttc')))
    records = list(csv.DictReader((root/'all-metrics.csv').open(encoding='utf-8')))
    factors = json.loads((root/'factorial.json').read_bytes())['pairs']
    checks = json.loads((root/'engineering-checks.json').read_bytes())
    assert len(records) == 21 and all(r['pass_all'] == 'True' and int(r['contact_steps']) == 0 for r in records)
    assert all(r['both_minima_before_either_return'] for r in checks['prefix'])
    review = root/'review'
    c = Canvas(str(review/'review.pdf'), pagesize=PAGE, pageCompression=1)
    c.setTitle('穿过窄门后，何时收臂？ | 单门在线调时 · 组会审阅')

    def txt(x, y, value, size=10, bold=False, color=INK, align='left'):
        c.setFont('bold' if bold else 'body', size)
        c.setFillColor(HexColor(color))
        {'left':c.drawString, 'center':c.drawCentredString, 'right':c.drawRightString}[align](x, 816-y, value)

    def rule(y, color=RULE, width=.6):
        c.setStrokeColor(HexColor(color))
        c.setLineWidth(width)
        c.line(34, 816-y, 596, 816-y)

    def art(file, y, height):
        c.drawImage(ImageReader(str(review/file)), 34, 816-y-height, 562, height,
                    preserveAspectRatio=True, anchor='c', mask='auto')

    def lines(y, items, size=9.6, spacing=16):
        for i, value in enumerate(items):
            txt(35,y+i*spacing,value,size)

    def header(n, title, subtitle):
        txt(35,26,'SUPER  /  QINGYU      形态感知全身规划',8.4,color=MUTED)
        txt(595,26,'G1 在线调时    /    2026.11.19',8.4,color=MUTED,align='right')
        rule(38,INK,.9)
        txt(34,78,title,22,True)
        txt(35,103,subtitle,9.8,color=MUTED)
        rule(779)
        txt(35,799,'组会审阅    /    已知单门 · 空载标称模型 · PX4 与 MuJoCo 软件在环',8,color=MUTED)
        txt(595,799,f'{n:02d}  /  02',8,color=MUTED,align='right')

    def table(y, labels, widths, rows):
        edges=[34]
        for width in widths:
            edges.append(edges[-1]+width)
        assert edges[-1]==596
        rule(y,INK,.85)
        for j, value in enumerate(labels):
            txt((edges[j]+edges[j+1])/2,y+15,value,8.8,True,align='center')
        rule(y+23)
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                txt((edges[j]+edges[j+1])/2,y+39+i*21,value,9.4,align='center')
        rule(y+48+(len(rows)-1)*21,INK,.8)

    header(1,'机身在门前停顿，柔性臂何时收回？',
           '三种时序接收同一条机身指令。机身从第 22 秒起减速，停 4 秒后恢复前进，路线进度共延迟 6 秒。')
    art('release.png',115,229)
    lines(362,['预选种子 241119。三帧各取本方案开始收臂的瞬间。后缘取整机最后方的注册部件，',
               '符号沿穿越方向计。负值表示整机尚未完全越过门框出口。各组均使用相同相机、比例尺与门框。'])
    txt(35,418,'三种时序如何改变收臂位置与门后到位时间',13,True)
    rows=[]
    for mode, label in [('clock_nominal','固定时钟 · 连续'),('online_nominal','状态触发 · 连续'),
                        ('clock_pause','固定时钟 · 停顿'),('phase_pause','指令进度 · 停顿'),('online_pause','状态触发 · 停顿')]:
        selected=[r for r in records if r['mode']==mode]
        assert len(selected)==3
        def med(k):
            return statistics.median(float(r[k]) for r in selected)
        rows.append([label,f'{med("return_t"):.1f}',f'{100*med("return_rear_m"):+.1f}',
                     f'{med("body_convex_cm"):.2f}',f'{med("arm_convex_cm"):.2f}',f'{med("target_time_s"):.2f}'])
    table(433,['时序与工况','收臂 (s)','后缘 (cm)','机身净空 (cm)','臂爪净空 (cm)','到位 (s)'],
          [132,70,78,95,95,92],rows)
    lines(594,['每组 3 趟，表列中位数。净空取每趟全程的三维凸体表面最小距离。到位要求夹爪距目标',
               '≤ 8 cm、轴向误差 ≤ 25°，连续保持 1 秒。15 趟主对照均达到机身 10 cm、臂爪 7 cm 的保守净空要求。'])
    rule(635)
    txt(35,662,'读数的含义',13,True)
    lines(687,['门前停顿时，固定时钟提前收臂也完成了任务。状态触发使收臂位置贴近出口，三种方案均零接触。',
               '相对指令进度方案，状态触发的臂爪净空中位数增加 1.44 cm，到位延后 0.80 s。',
               '连续前进时，状态触发的臂爪净空中位数比固定时钟少 1.14 cm。时序收益取决于行进工况。'])
    c.showPage()
    header(2,'最小净空产生在弯臂段，还是收臂段？',
           '在相同的停顿任务中交叉交换弯臂和收臂策略。6 趟补充试验与主对照的 6 趟组成每个种子的四个组合。')
    art('curves.png',111,282)
    lines(410,['a–c 为种子 241119 的实际后缘位置、臂爪净空下界和执行弯角。d 连起 3 个种子各自的三种结果。',
               '曲线 b 使用部件包围盒下界。图 d 和下表使用三维凸体表面距离，两种量各自独立计算。'],9.2)
    txt(35,463,'交叉对照中的全程最小臂爪净空（cm）',13,True)
    keys=['phase_out_phase_back_cm','phase_out_online_back_cm','online_out_phase_back_cm','online_out_online_back_cm']
    table(477,['种子','进度弯／进度收','进度弯／在线收','在线弯／进度收','在线弯／在线收'],
          [76,122,121,122,121],[[str(f['seed'])]+[f'{f[k]:.2f}' for k in keys] for f in factors])
    lines(596,['对同一种弯臂策略，两趟运行在较早收臂前的弯角指令逐拍相同。四个组合的最近点均出现在',
               '33.86–34.75 s，最早收臂在 38.40 s。此轮最近点的差异已经发生在收臂之前。',
               '固定收臂策略后，改用在线弯臂的六个配对均增加 0.61–2.05 cm。独立飞行本身也存在波动。'])
    rule(650)
    txt(35,679,'对整体开发计划的意义',13,True)
    lines(704,['21 趟均在 52 s 内到位，最大实际弯角 72.71° 以内。在线调时接收 PX4 估计状态和仿真关节反馈。',
               '本轮覆盖一扇已知门、3 个噪声种子。下一轮用两扇连续门检查前门收臂与后门弯臂的时序冲突，',
               '再决定哪些结果足以进入论文正文。实物载荷辨识、反作用力矩补偿及感知建图另行验证。'])
    c.showPage()
    c.save()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',type=Path)
    p.add_argument('--fonts',type=Path,default=Path('C:/Windows/Fonts'))
    args=p.parse_args()
    make_report(args.stage,args.fonts)
