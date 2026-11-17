from pathlib import Path
import fitz
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path(__file__).resolve().parents[1];F=R/'figures'
# Recompose the original side and 3-D views without altering their vector paths.
s=fitz.open(F/'fig04_measured_whole_body_trajectory.pdf');g=fitz.open(F/'geometry_clean.pdf')
for text in ['Closed-loop simulated path and arm snapshots (3-D)', '(c)']:
    for rect in s[0].search_for(text): s[0].add_redact_annot(rect,fill=False)
s[0].apply_redactions(images=0,graphics=0,text=0)
out=fitz.open();p=out.new_page(width=516,height=250)
p.insert_text((12,13),'(a) Spatial passage and arm configurations',fontsize=9,fontname='hebo')
p.insert_text((271,13),'(b) Side view and straight-arm counterfactual',fontsize=9,fontname='hebo')
p.show_pdf_page(fitz.Rect(0,24,251,247),s,0,clip=fitz.Rect(305,21,534,345))
p.show_pdf_page(fitz.Rect(258,42,516,226),g,0)
out.save(F/'passage_restored.pdf')
# The source pixels, residual display limits and time intervals are preserved.
# New vector axes improve readability at two-column journal scale. This is
# artwork composition, not recovery or interpolation of experimental samples.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'axes.linewidth':.5})
fig,ax=plt.subplots(3,2,figsize=(7.16,5.45));fig.subplots_adjust(left=.075,right=.985,bottom=.14,top=.84,hspace=.61,wspace=.27)
specs=[('fig_sensor_theta_fusion.png',[(45,77,1278,297),(45,361,1278,583),(45,647,1278,867)],[(100,560,-5,110),(100,560,-12,12),(350,455,-6,12)]),('fig_sensor_phi_fusion.png',[(49,77,1268,295),(49,358,1268,578),(49,640,1268,860)],[(80,700,-115,105),(80,700,-18,18),(340,480,-24,24)])]
for col,(name,crops,limits) in enumerate(specs):
 im=Image.open(F/name)
 for row,(crop,lim) in enumerate(zip(crops,limits)):
  a=ax[row,col];a.imshow(im.crop(crop),extent=lim,origin='upper',aspect='auto',interpolation='none',rasterized=True)
  a.set_xlim(lim[0],lim[1]);a.set_ylim(lim[2],lim[3]);a.set_xlabel('Time (s)',labelpad=1)
  a.set_ylabel(['Angle (deg)','Error (deg)','Error (deg)'][row] if col==0 else ['Angle (deg)','Circular error (deg)','Circular error (deg)'][row],labelpad=2)
  a.spines[['top','right']].set_visible(False)
  a.set_title(['Full recorded trajectory','Full recorded residual','Visual-degradation detail'][row],loc='left',pad=5)
  if row==1:a.set_yticks([-12,0,12] if col==0 else [-18,0,18])
  if row==0:a.set_yticks([0,50,100] if col==0 else [-100,0,100])
  if row==2:a.set_yticks([-6,0,6,12] if col==0 else [-24,-12,0,12,24])
fig.text(.29,.971,r'Bending magnitude $\theta$',ha='center',fontsize=10,weight='bold')
fig.text(.77,.971,r'Bending plane $\phi$',ha='center',fontsize=10,weight='bold')
fig.text(.29,.943,'Command-derived reference',ha='center',fontsize=7.5)
fig.text(.77,.943,'Normal-ORB-fitted reference',ha='center',fontsize=7.5)
handles=[Line2D([0],[0],color=c,lw=1.3,ls=ls,label=l) for c,ls,l in [('black','-','Reference'),('#1677c8',':','ORB'),('#f47920','-','Strain'),('#8c8c8c','--','Static mixture'),('#b21e2b','-','Fusion')]]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.923),ncol=5,frameon=False,fontsize=7.5,handlelength=2,columnspacing=1.5)
fig.text(.5,.06,'Residual panels retain the source display limits: failed ORB errors exceed the visible range.',ha='center',fontsize=7.5,color='#444444')
fig.text(.5,.031,'Complete source images and printed statistics are retained in Supplementary Figs. S4 and S5.',ha='center',fontsize=7.5,color='#444444')
fig.savefig(F/'sensing_restored.png',dpi=450)
