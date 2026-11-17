from pathlib import Path
from math import sin,cos,pi
from html import escape
import cairosvg
R=Path(__file__).resolve().parent
S=[]
ink='#243645'; muted='#61717F'; blue='#246B98'; teal='#207F79'; amber='#B76C27'; light='#D9E1E7'
def emit(s):S.append(s)
def text(x,y,t,fs=22,color=ink,weight=400,anchor='start'):
 emit(f'<text x="{x}" y="{y}" font-family="Arial,DejaVu Sans,sans-serif" font-size="{fs}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(t)}</text>')
def path(d,stroke=ink,w=2,fill='none',dash=None,opacity=1):emit(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
def line(x1,y1,x2,y2,c=ink,w=2,dash=None):path(f'M{x1},{y1} L{x2},{y2}',c,w,dash=dash)
def rect(x,y,w,h,fill='white',stroke=light,rx=8,sw=1.6):emit(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def ell(x,y,rx,ry,fill,stroke=ink,sw=1.5):emit(f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def arrow(points,c=ink,w=2.3,dash=None):
 d='M'+' L'.join(f'{x},{y}' for x,y in points)
 emit(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{w}" stroke-linejoin="round"'+(f' stroke-dasharray="{dash}"' if dash else '')+f' marker-end="url(#{"blue" if c==blue else "teal" if c==teal else "amber" if c==amber else "ink"})"/>')
def title(x,y,num,t,c,w):
 rect(x,y-24,33,33,c,c,5);text(x+16.5,y+1,num,21,'white',700,'middle');text(x+48,y+2,t,26,ink,700);line(x,y+23,x+w,y+23,light,1.5)
def card(x,y,w,h,title_,sub,c=blue):
 rect(x,y,w,h,'#FFFFFF',light,8)
 rect(x,y,5,h,c,c,2,0)
 text(x+22,y+34,title_,24,ink,700)
 for j,l in enumerate(sub):text(x+22,y+64+j*27,l,21,muted)
def step(y,n,title_,subs,h=104,c=amber):
 x=978;w=624
 rect(x,y,w,h,'#FCFDFE',light,8)
 ell(x+29,y+31,15,15,'#F4E8DC','none',0);text(x+29,y+38,n,20,c,700,'middle')
 text(x+57,y+39,title_,24,ink,700)
 for j,l in enumerate(subs):text(x+57,y+70+j*26,l,21,muted)
emit('<svg xmlns="http://www.w3.org/2000/svg" width="1640" height="930" viewBox="0 0 1640 930">')
emit('''<defs>
<linearGradient id="metal"><stop stop-color="#9EA9B3"/><stop offset=".35" stop-color="#EEF1F3"/><stop offset=".64" stop-color="#B7C2CB"/><stop offset="1" stop-color="#717E8A"/></linearGradient>
<linearGradient id="carbon"><stop stop-color="#121E28"/><stop offset=".5" stop-color="#46515B"/><stop offset="1" stop-color="#17232E"/></linearGradient>
<linearGradient id="lens" x2="0" y2="1"><stop stop-color="#64A0C1"/><stop offset="1" stop-color="#102936"/></linearGradient>
''')
for name,c in [('ink',ink),('blue',blue),('teal',teal),('amber',amber)]:emit(f'<marker id="{name}" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L9,4.5 L0,9 Z" fill="{c}"/></marker>')
emit('</defs>');rect(0,0,1640,930,'white','none',0,0)
title(24,40,'a','Aerial continuum platform',ink,400)
title(467,40,'b','Visual–strain morphology',blue,448)
title(978,40,'c','Rolling whole-body planning',amber,624)
line(445,85,445,835,light,1);line(947,85,947,835,light,1)
# Stylized technical reconstruction from the supplied multirotor photographs/CAD.
# Four motors/propellers; underbody actuation module; exactly one PCC section.
for x,y in [(118,148),(289,148),(72,222),(337,222)]:
 line(207,191,x,y,'#283640',12)
 ell(x,y+10,19,8,'#58646E');rect(x-17,y-8,34,18,'url(#carbon)',ink,3)
 ell(x,y-8,17,6,'#9CA7AF')
 emit(f'<g transform="translate({x},{y-16}) rotate({-10 if y<180 else 12})">')
 path('M-65,0 Q-28,-12 0,-3 Q28,-12 65,0 Q28,9 0,3 Q-28,9 -65,0',ink,1.3,'#52606B')
 ell(0,0,5,5,'#A9B5BE');emit('</g>')
path('M147,165 L241,153 L269,174 L171,188 Z',ink,2,'#46525D')
path('M147,165 L171,188 L171,236 L147,215 Z',ink,2,'#263540')
path('M171,188 L269,174 L269,222 L171,236 Z',ink,2,'#56636E')
for x,y in [(151,169),(251,177),(173,221),(253,215)]:line(x,y,x,y+38,'#657681',3)
rect(177,198,69,14,'#325663',ink,1)
for i in range(8):line(181+i*8,198,181+i*8,207,'#BFA976',1.3)
# Dome and avionics silhouette; no unverified function label for the top sensor.
rect(182,131,53,27,'url(#metal)',ink,3)
for i in range(9):line(185+i*5,136,185+i*5,157,'#74828E',1)
path('M185,131 a23,23 0 0 1 46,0 Z',ink,1.6,'url(#lens)')
path('M178,153 Q165,95 208,103 Q248,112 241,153',ink,4)
# Underbody structural bracket and cylindrical actuator.
for x in [178,230]:line(x,235,x,267,ink,4)
ell(204,269,47,11,'#CBD2D7');rect(157,269,94,54,'url(#metal)',ink,1)
ell(204,323,47,11,'#A2ADB5')
for x in range(169,241,12):line(x,282,x-7,310,'#536572',2)
rect(186,330,36,23,'url(#metal)',ink,4)
# Analytic PCC centerline. Thin rib ellipses are perpendicular to the tangent.
L=172;theta=pi/3;start=(204,354)
def arc(s,offset=0):
 u=theta*s/L
 return start[0]+L/theta*(1-cos(u))+offset*cos(u),start[1]+L/theta*sin(u)-offset*sin(u)
pts=[arc(L*i/100) for i in range(101)]
d='M'+' L'.join(f'{x:.3f},{y:.3f}' for x,y in pts)
path(d,'#1A2731',31);path(d,'#66717A',24);path(d,'#313E49',17)
for i in range(30):
 ss=L*i/29;x,y=arc(ss);u=theta*ss/L
 emit(f'<g transform="translate({x},{y}) rotate({-u*180/pi})">')
 ell(0,0,16,3.8,'none','#AEB8C0',1.4);emit('</g>')
# Two visible gauges; cross section below explicitly identifies all four channels.
for off in [-9,9]:
 ps=[arc(L*t/100,off) for t in range(18,63)]
 path('M'+' L'.join(f'{x:.3f},{y:.3f}' for x,y in ps),'#C79045',3.1)
line(204,356,204,522,'#BEC8D0',1.4,'5 5')


x,y=arc(L)
emit(f'<g transform="translate({x},{y}) rotate(-60)">')
rect(-19,-3,38,21,'url(#metal)',ink,3)
rect(-40,15,80,25,'url(#metal)',ink,8)
rect(-34,20,68,14,'#23313C',ink,6)
for xx in [-24,-3,23]:ell(xx,27,5,5,'url(#lens)','#A5B6C2',.7)
rect(-17,40,34,13,'#4C5965',ink,3)
for xx in [-15,0,15]:
 path(f'M{xx},51 L{xx*1.8},69 L{xx*1.5},88 L{xx*.9},78 L{xx*.9},65 Z',ink,1.2,'#73818B')
emit('</g>')
# Mechanical callouts: no numerical dimensions inferred from screenshots.
text(24,109,'Four-rotor UAV',22,ink,700);line(155,113,176,151,muted,1)
text(23,290,'Underbody',21);text(23,315,'actuation',21);line(130,304,156,292,muted,1)
text(24,400,'Single PCC',22,ink,700);text(24,426,'bending section',21);line(156,414,212,406,muted,1)
text(25,531,'Rigid distal assembly',22,ink,700);text(25,558,'D435 + gripper',22);line(222,529,x+25,y+14,muted,1)
# Sensor cross section and clear distinction between channels and segments.
ell(89,646,37,37,'#F2F5F7','#687B8C',1.5)
for i in range(4):
 a=i*pi/2;gx=89+37*cos(a);gy=646+37*sin(a)
 rect(gx-5,gy-11,10,22,'#C79045','white',2,1)
text(151,627,'Four strain channels',22,ink,700)
text(151,657,'Circumferential sensing',20,muted)
text(151,685,'One continuum section',20,muted)
text(24,746,'Mechanical relationships',22,ink,700)
text(24,777,'UAV → actuator → bending section',20,muted)
text(24,807,'→ rigid camera / gripper assembly',20,muted)
# Offline training and online observations.
rect(687,105,226,125,'#F5F3FA','#CBC2DB',8)
text(706,137,'OFFLINE',18,'#756087',700)
text(706,169,'Mocap + strain',23,ink,700)
text(706,201,'Supervised training',20,muted)
arrow([(800,230),(800,288)],blue,2.1,'6 4');text(803,266,'fixed model',18,muted)
card(467,173,199,98,'UAV pose',['Base motion'],blue)
arrow([(566,271),(566,288)],blue)
# Parallel sensor branches.
rect(467,289,199,181,'#F5F9FC','#B8CDDD',8)
text(486,324,'EYE-IN-HAND',18,blue,700)
text(486,357,'D435 / RGB-D',23,ink,700)
text(486,387,'ORB-SLAM3',22)
text(486,420,'Base-motion',21,muted)
text(486,446,'compensation',21,muted)
rect(687,289,226,181,'#F5F9FC','#B8CDDD',8)
text(707,324,'STRAIN',18,blue,700)
text(707,357,'Four channels',23,ink,700)
text(707,389,'Fixed predictor',22)
text(707,421,'Morphology',21,muted)
text(707,447,'observation',21,muted)
arrow([(566,470),(566,521)],blue);arrow([(800,470),(800,521)],blue)
text(482,502,'tracking quality',18,blue);text(724,502,'strain estimate',18,blue)
card(481,522,418,130,'Confidence-adaptive fusion',['Angular / circular weighting','Temporal filtering'],blue)
arrow([(690,652),(690,691)],blue)
rect(532,692,318,66,'#EAF3F8','#9EBACD',7)
text(691,721,'MORPHOLOGY STATE',18,blue,700,'middle')
text(691,747,'θ, φ  /  bending vector b',23,ink,500,'middle')
text(481,805,'Vision degrades → strain gains weight',20,blue)
text(481,832,'No online retraining',20,muted)
# Rolling pipeline: the planning region has the strongest visual hierarchy.
card(978,100,624,85,'SLAM occupancy + global reference',[],amber)
text(1000,164,'Local map and reference-path input',21,muted)
arrow([(1290,185),(1290,217)],amber)
step(218,'1','Local horizon & geometric seed',['Current UAV state + morphology; coupled seed'])
arrow([(1290,322),(1290,350)],amber)
step(351,'2','Joint trajectory & timing refinement',['UAV position + yaw + θ, φ + segment times','Finite computation budget'],125)
arrow([(1290,476),(1290,504)],amber)
step(505,'3','Independent dense validation',['Whole-body clearance · CoM · tilt / thrust proxy','Morphology-rate limits'],125,teal)
arrow([(1290,630),(1290,658)],teal)
step(659,'4','Handoff authorization',['C1 direct gate  /  quintic C2 bridge','Bridge must pass validation'],125,teal)
arrow([(1290,784),(1290,812)],teal)
rect(978,813,624,59,'#EAF5F2','#A3C5BC',8)
text(1290,850,'Validated buffer → PX4 / arm control',25,teal,700,'middle')
# A separate interface makes the evaluation boundary explicit.
arrow([(850,725),(929,725),(929,303),(978,303)],blue,2.3,'7 5')

emit('<text x="1634" y="475" transform="rotate(-90 1634 475)" font-family="Arial" font-size="17" fill="#61717F">Reject / replan</text>')
# Retry route is confined to the right outside gutter; no unchecked execution.
arrow([(1602,553),(1623,553),(1623,270),(1602,270)],muted,1.8)
path('M1602,715 L1623,715 L1623,553',muted,1.8)
# The operating loop returns authorized commands to the hardware.
arrow([(978,842),(945,842),(945,893),(410,893),(410,578),(365,540)],teal,2.2)
text(568,885,'Authorized vehicle and arm commands',20,teal)
text(24,926,'Dashed blue link: estimator interface; planning benchmark used modeled morphology through the same interface.',20,muted)
emit('</svg>')
svg='\n'.join(S);(R/'overall_framework.svg').write_text(svg)
cairosvg.svg2pdf(bytestring=svg.encode(),write_to=str(R/'overall_framework.pdf'),output_width=516,output_height=930/1640*516)
cairosvg.svg2png(bytestring=svg.encode(),write_to=str(R/'overall_framework.png'),output_width=3280,output_height=1860)
print('Created SVG, vector PDF and PNG preview.')
