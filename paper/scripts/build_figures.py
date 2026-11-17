"""Recompose supplied figures without changing plotted samples or numeric results.

Requires PyMuPDF, Pillow, and reportlab. Original assets remain untouched.
Run from any directory: python scripts/build_figures.py
"""
from pathlib import Path
import math
import fitz
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / 'figures'

def copy_pdf(source, dest, replacements=None):
    doc = fitz.open(F/source)
    page = doc[0]
    changed = []
    for old,new in (replacements or {}).items():
        for rect in page.search_for(old):
            # This operation changes text only: images and vector paths are retained.
            page.add_redact_annot(rect, fill=False)
            changed.append((rect, new))
    if changed:
        page.apply_redactions(images=0, graphics=0, text=0)
        for rect,txt in changed:
            fs=min(8.0,rect.height*.7)
            while fitz.get_text_length(txt,fontname='helv',fontsize=fs)>rect.width and fs>5:
                fs-=.2
            page.insert_text((rect.x0,rect.y1-2),txt,fontsize=fs,fontname='helv')
    doc.save(F/dest,garbage=4,deflate=True)
    doc.close()

copy_pdf('fig01_success_and_failure_overview.pdf','outcomes_revised.pdf')
copy_pdf('fig02_execution_quality_distributions.pdf','quality_revised.pdf')
copy_pdf('fig05_rolling_horizon_updates.pdf','rolling_revised.pdf')
copy_pdf('fig07_handoff_continuity.pdf','handoff_revised.pdf',{
    'Closed-loop simulated command continuity around a validated handoff':
    'Command continuity around a revalidated handoff'})

# Enlarge the scientifically relevant side view; all other views remain in S6.
src=fitz.open(F/'fig04_measured_whole_body_trajectory.pdf')
clip=fitz.Rect(30,187,274,345)
out=fitz.open(); pg=out.new_page(width=516,height=334)
pg.show_pdf_page(fitz.Rect(0,0,516,334),src,0,clip=clip)
out.save(F/'geometry_revised.pdf',garbage=4,deflate=True)

# Retain the entire displayed time interval in each angle panel. Residual/detail
# panels are retained intact in the supplementary document, with range disclosure.
out=fitz.open(); pg=out.new_page(width=516,height=292)
for idx,(name,heading) in enumerate([
    ('fig_sensor_theta_fusion.png','(a) Bending magnitude: command-derived evaluation reference'),
    ('fig_sensor_phi_fusion.png','(b) Bending plane: normal-ORB-fitted evaluation reference')]):
    img=Image.open(F/name); w,h=img.size
    imdoc=fitz.open(); ip=imdoc.new_page(width=w,height=h)
    ip.insert_image(ip.rect,filename=str(F/name))
    y=idx*145
    pg.insert_text((7,y+10),heading,fontsize=8.4,fontname='hebo')
    labels=[('Evaluation reference',(0,0,0)),('ORB',(0,.38,.76)),('Strain',(1,.36,0)),('Fusion',(.75,0,.13))]
    x=76
    for label,col in labels:
        pg.draw_line((x,y+22),(x+14,y+22),color=col,width=1.2)
        pg.insert_text((x+17,y+25),label,fontsize=7.5,fontname='helv')
        x += 118 if label=='Evaluation reference' else 75
    c=fitz.Rect(0,h*(.083 if idx else .077),w,h*.356)
    pg.show_pdf_page(fitz.Rect(0,y+32,516,y+132),imdoc,0,clip=c,keep_proportion=False)
    pg.insert_text((246,y+143),'Time (s)',fontsize=7.5,fontname='helv')
out.save(F/'sensing_revised.pdf',garbage=4,deflate=True)

# Exact vector schematic. All flexible centerlines are analytic constant-curvature
# arcs; the camera/tool has a fixed rotation with respect to the distal tangent.
c=canvas.Canvas(str(F/'framework_revised.pdf'),pagesize=(516,250))
ink=HexColor('#243746'); blue=HexColor('#176ca4'); teal=HexColor('#16756c')
orange=HexColor('#b45812'); gray=HexColor('#e6edf1')
def text(x,y,t,size=8,color=ink,bold=False):
    c.setFillColor(color);c.setFont('Helvetica-Bold' if bold else 'Helvetica',size)
    c.drawCentredString(x,y,t)
def box(x,y,w,h,lines,color=blue):
    c.setStrokeColor(color);c.setFillColor(white);c.setLineWidth(.7)
    c.roundRect(x,y,w,h,3,stroke=1,fill=1)
    for j,line in enumerate(lines):text(x+w/2,y+h/2+(len(lines)-1)*4.7-j*9.4,line,8,color)
def arrow(points,color=ink,dash=False):
    c.setStrokeColor(color);c.setFillColor(color);c.setLineWidth(.85)
    c.setDash(2,2) if dash else c.setDash()
    p=c.beginPath();p.moveTo(*points[0])
    for x,y in points[1:]:p.lineTo(x,y)
    c.drawPath(p);c.setDash()
    x,y=points[-1];a,b=points[-2];ang=math.atan2(y-b,x-a)
    p=c.beginPath();p.moveTo(x,y)
    p.lineTo(x-4*math.cos(ang-.45),y-4*math.sin(ang-.45))
    p.lineTo(x-4*math.cos(ang+.45),y-4*math.sin(ang+.45));p.close();c.drawPath(p,fill=1,stroke=0)

for x,w,title,color in [(2,153,'(a) Morphology feedback',blue),(170,174,'(b) Rolling planning',orange),(359,155,'(c) Transition / execution',teal)]:
    c.setStrokeColor(color);c.setLineWidth(.6);c.roundRect(x,20,w,226,4,stroke=1,fill=0)
    text(x+w/2,232,title,9,color,True)

# UAV in side projection, suspended mount, one circular arc, distal camera/gripper.
c.setStrokeColor(ink);c.setFillColor(gray);c.setLineWidth(1.2)
c.line(35,205,111,205);c.ellipse(26,202,48,209,fill=1);c.ellipse(99,202,121,209,fill=1)
c.roundRect(61,198,26,15,3,fill=1);c.line(74,198,74,189)
L=42;theta=math.radians(60);pts=[]
for i in range(81):
    u=theta*i/80
    pts.append((74+L/theta*(1-math.cos(u)),189-L/theta*math.sin(u)))
c.setStrokeColor(blue);c.setLineWidth(5)
p=c.beginPath();p.moveTo(*pts[0])
for xy in pts[1:]:p.lineTo(*xy)
c.drawPath(p)
x,y=pts[-1];c.saveState();c.translate(x,y);c.rotate(60)
c.setFillColor(gray);c.setStrokeColor(ink);c.setLineWidth(.8)
c.rect(-12,-9,24,7,stroke=1,fill=1)
for xx in [-8,0,8]:c.circle(xx,-6,1.5,stroke=1,fill=0)
arrow([(0,-9),(0,-20)],blue)
c.line(-7,-10,-7,-17);c.line(7,-10,7,-17);c.restoreState()
text(43,171,'Single PCC arc',7.2)
text(51,147,'D435 + tool',7.2)
# Sensor cross-section, four orthogonal gauges.
c.setStrokeColor(ink);c.setLineWidth(.7);c.circle(36,121,9,stroke=1,fill=0)
for i in range(4):
    a=i*math.pi/2;xx=36+9*math.cos(a);yy=121+9*math.sin(a)
    c.setFillColor(orange);c.circle(xx,yy,2,stroke=0,fill=1)
text(98,123,'Four strain channels',7.1)
text(98,112,'90-degree spacing',7.1)
box(10,72,65,26,['Strain model'],blue)
box(82,65,65,33,['End-eye vision','Base-motion','compensation'],blue)
text(45,103,'Mocap (offline)',7.1,orange)
arrow([(43,102),(43,98)],orange,True)
box(18,32,130,25,['Confidence-adaptive fusion'],blue)
arrow([(42,72),(42,57)],blue);arrow([(114,65),(114,57)],blue)

box(184,187,146,30,['Occupancy map / global reference'],orange)
box(184,143,146,29,['Geometry-guided whole-body seed'],orange)
box(184,96,146,32,['Joint spline / timing refinement','Position, yaw, bending, duration'],orange)
box(184,47,146,31,['Dense whole-body validation','Geometry, CoM, motion limits'],orange)
for ya,yb in [(187,172),(143,128),(96,78)]:arrow([(257,ya),(257,yb)],orange)
arrow([(148,44),(162,44),(162,157),(184,157)],blue,True)

box(373,180,126,36,['Phase match / direct gate','Finite-tolerance C1 matching'],teal)
box(373,125,126,33,['If needed: quintic bridge','Match derivatives; revalidate'],teal)
box(373,77,126,29,['Validated command buffer'],teal)
box(373,32,126,27,['PX4 / arm execution'],teal)
arrow([(330,62),(351,62),(351,198),(373,198)],teal)
arrow([(436,180),(436,158)],teal)
arrow([(436,125),(436,106)],teal)
arrow([(436,77),(436,59)],teal)
arrow([(499,198),(507,198),(507,92),(499,92)],teal)
text(491,165,'pass',6.8,teal)
arrow([(436,32),(436,10),(77,10),(77,32)],ink)
text(253,12,'Updated UAV state and morphology interface',7.1)
c.save()
print('Revised figures generated.')
