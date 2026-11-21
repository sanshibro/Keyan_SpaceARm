"""Publish native paths for analytical layers and color-traced CAD shading."""
import base64
import copy
import io
import os
import re
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import vtracer

from m0_data import OUT, QA, WORK, sha_file, write_json

if os.name == 'nt':
    lib=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
    if lib.is_dir():
        os.environ['CAIROCFFI_DLL_DIRECTORIES']=str(lib)
        os.environ['PATH']=str(lib)+os.pathsep+os.environ['PATH']
        cairo_search=os.add_dll_directory(str(lib))
import cairosvg
import pymupdf

NS='http://www.w3.org/2000/svg'
XL='http://www.w3.org/1999/xlink'
ET.register_namespace('',NS)
ET.register_namespace('xlink',XL)
S=lambda name:'{'+NS+'}'+name
STEMS=('gate_hero_v2','gate_insets_v2','gate_four_metrics_v2')
CACHE=WORK/'vector'
CACHE.mkdir(exist_ok=True)


def trace_layers(root,stem):
    layers=[]
    for parent in root.iter():
        for node in list(parent):
            if node.tag!=S('image'):continue
            href=node.get('{'+XL+'}href',node.get('href',''))
            bitmap=Image.open(io.BytesIO(base64.b64decode(href.partition(',')[2]))).convert('RGBA')
            width,height=bitmap.size
            white=Image.new('RGBA',bitmap.size,'white')
            white.alpha_composite(bitmap)
            # Keep film-sized boundaries and suppress Monte Carlo sub-pixel oscillations.
            raster=white.convert('RGB');raster.thumbnail((1880,1250),Image.Resampling.LANCZOS)
            sw,sh=raster.size
            file=CACHE/(stem+'-'+str(len(layers))+'.png')
            raster.save(file);out=file.with_suffix('.svg')
            vtracer.convert_image_to_svg_py(str(file),str(out),colormode='color',hierarchical='stacked',
                mode='spline',filter_speckle=3,color_precision=8,layer_difference=4,corner_threshold=55,
                length_threshold=3.2,max_iterations=10,splice_threshold=43,path_precision=2)
            traced=ET.parse(out).getroot()
            prot={k:v for k,v in node.attrib.items() if k not in
                  ('x','y','width','height','preserveAspectRatio','href','{'+XL+'}href','style')}
            group=ET.Element(S('g'),prot)
            group.set('data-origin','color-traced CAD lighting')
            slot=ET.SubElement(group,S('g'),{'transform':'translate({} {}) scale({:.11g} {:.11g})'.format(
                node.get('x','0'),node.get('y','0'),float(node.get('width'))/sw,float(node.get('height'))/sh)})
            for child in list(traced):
                if child.tag in (S('g'),S('path')):slot.append(child)
            idx=list(parent).index(node);parent.remove(node);parent.insert(idx,group)
            layers.append(dict(input_size=[width,height],trace_size=[sw,sh],paths=len(list(slot.iter(S('path'))))))
    return layers


def svg_pdf(stem,root):
    svg=OUT/(stem+'.svg');pdf=OUT/(stem+'.pdf')
    ET.ElementTree(root).write(svg,encoding='utf-8',xml_declaration=True)
    cairosvg.svg2pdf(url=str(svg),write_to=str(pdf))
    doc=pymupdf.open(pdf)
    if len(doc)!=1 or doc[0].get_images():raise ValueError('Unexpected raster XObject: '+stem)
    return doc


def score(stem,doc):
    original=Image.open(OUT/(stem+'.png')).convert('RGB')
    width=1300;h=round(width*original.height/original.width)
    pix=doc[0].get_pixmap(matrix=pymupdf.Matrix(width/doc[0].rect.width,width/doc[0].rect.width),alpha=False)
    pdf=Image.frombytes('RGB',(pix.width,pix.height),pix.samples).resize((width,h),Image.Resampling.LANCZOS)
    pdf.save(QA/(stem+'-vector-proof.jpg'),quality=91)
    ref=np.array(original.resize((width,h),Image.Resampling.LANCZOS),dtype='f4')
    error=np.abs(np.asarray(pdf,dtype='f4')-ref).mean(axis=2)
    if error.mean()>12:raise ValueError(f'{stem} differs after tracing: {error.mean():.2f} / 255')
    return {'mean_absolute_RGB_difference_255':round(float(error.mean()),2),
            'page_points':list(doc[0].rect)[2:],'pdf_sha256':sha_file(OUT/(stem+'.pdf'))}


def place(root,src,x,y,w,prefix):
    view=list(map(float,src.get('viewBox').split()))
    data=copy.deepcopy(src)
    ids={el.get('id'):prefix+el.get('id') for el in data.iter() if el.get('id')}
    for elem in data.iter():
        for key,value in list(elem.attrib.items()):
            if key=='id':elem.set(key,ids[value]);continue
            if key in ('href','{'+XL+'}href') and value.startswith('#') and value[1:] in ids:
                elem.set(key,'#'+ids[value[1:]]);continue
            if 'url(#' in value:elem.set(key,re.sub(r'url\(#([^)]+)\)',lambda m:'url(#'+ids.get(m[1],m[1])+')',value))
    g=ET.SubElement(root,S('g'),{'transform':f'translate({x:.5f} {y:.5f}) scale({w/view[2]:.8f})'})
    for child in data:
        if child.tag!=S('metadata'):g.append(child)
    return w*view[3]/view[2]


def main():
    roots=[];meta={}
    for stem in STEMS:
        src=ET.parse(WORK/(stem+'-layers.svg')).getroot()
        layers=trace_layers(src,stem)
        if list(src.iter(S('image'))):raise ValueError('Untraced bitmap')
        doc=svg_pdf(stem,src)
        meta[stem]={**score(stem,doc),'CAD_layers':layers,'native_text':len(list(src.iter(S('text')))),
                    'all_paths':len(list(src.iter(S('path')))),'pdf_images':len(doc[0].get_images())}
        roots.append(src);doc.close()
        print(stem,meta[stem]['mean_absolute_RGB_difference_255'],flush=True)
    page=ET.Element(S('svg'),{'version':'1.1'})
    width=1020.;gap=7.;half=(width-gap)/2
    hero_h=place(page,roots[0],0,0,width,'hero_')
    bh=max(place(page,roots[1],0,hero_h+gap,half,'inset_'),place(page,roots[2],half+gap,hero_h+gap,half,'plot_'))
    h=hero_h+gap+bh;page.set('width',f'{width}pt');page.set('height',f'{h:.3f}pt');page.set('viewBox',f'0 0 {width} {h:.3f}')
    bg=ET.Element(S('rect'),{'width':str(width),'height':str(h),'fill':'white'});page.insert(0,bg)
    doc=svg_pdf('gate_main_v2',page)
    scale=3.4
    rgb=Image.new('RGB',(round(width*scale),round(h*scale)),'white')
    for stem,xx,yy,pw in zip(STEMS,(0,0,half+gap),(0,hero_h+gap,hero_h+gap),(width,half,half)):
        im=Image.open(OUT/(stem+'.png')).convert('RGB');destw=round(pw*scale)
        im=im.resize((destw,round(destw*im.height/im.width)),Image.Resampling.LANCZOS)
        rgb.paste(im,(round(xx*scale),round(yy*scale)))
    rgb.save(OUT/'gate_main_v2.png',dpi=(300,300))
    meta['gate_main_v2']={'pdf_images':len(doc[0].get_images()),'png_size':list(rgb.size),
                           'pdf_sha256':sha_file(OUT/'gate_main_v2.pdf')}
    write_json(QA/'vector-verification.json',meta)
if __name__=='__main__':main()
