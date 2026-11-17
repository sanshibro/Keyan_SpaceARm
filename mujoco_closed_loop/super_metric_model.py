"""SUPER CAD-registered development model with explicit collision primitives.

This is a new configuration. Old runs and their model factory are unchanged.
The rigid fit is conservative; mass, actuator and attachment parameters remain
development assumptions until physical identification and installation checks.
"""
from pathlib import Path
import json
import numpy as np
from xml.etree.ElementTree import fromstring, SubElement as S, tostring
from model import model_xml

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'revision/data/super_metric_geometry.json'

def fmt(v): return ' '.join('%.9g'%x for x in v)

def build_super_metric(fixed=True,segments=8):
    info=json.loads(MANIFEST.read_text())
    root=fromstring(model_xml(fixed=fixed,segments=segments,profile='qingyu',
                            camera_profile='original',nav_camera_profile='front',super_visual=False))
    root.set('model',info['platform_id'])
    asset=root.find('asset');world=root.find('worldbody');u=world.find("body[@name='uav']")
    visual=root.find('visual');S(visual,'global',offwidth='1600',offheight='1200',orthographic='true',fovy='.80')
    S(visual,'headlight',ambient='.55 .55 .55',diffuse='.65 .65 .65',specular='.15 .15 .15')
    S(visual,'quality',shadowsize='2048',offsamples='4')
    S(visual,'rgba',haze='1 1 1 1')
    S(asset,'texture',name='white_studio',type='skybox',builtin='flat',rgb1='1 1 1',width='256',height='1536')
    # Existing masses are retained explicitly while replacing the old collision
    # block. New massless proxies cannot silently change inferred inertia.
    for g in u.findall('geom'):
        g.set('contype','0');g.set('conaffinity','0');g.set('rgba','0 0 0 0')
    for site in u.findall('site'):
        if site.get('name','').startswith('rotor'):
            i=int(site.get('name')[5:]);site.set('pos',fmt(info['rotors'][i]['axis_center_m']));site.set('rgba','0 0 0 0')
    colors={'carbon':'.29 .33 .36 1','motor':'.38 .40 .42 1','prop':'.24 .28 .31 1',
        'metal_lidar':'.48 .52 .55 1','Lidar':'.74 .78 .80 1','lander':'.31 .36 .39 1',
        'PCB':'.29 .43 .38 1','metal':'.57 .60 .62 1','yellow':'.78 .62 .30 1',
        'green':'.36 .54 .40 1','Led_front':'.33 .58 .62 1','Led_back':'.65 .38 .34 1',
        'mars_logo':'.78 .80 .81 1','usb':'.64 .66 .68 1'}
    for name,p in info['parts'].items():
        if name=='lander':continue # replaced by the user's long landing assembly
        S(asset,'mesh',name='metric_'+name,file=str(ROOT/p['file']))
        S(u,'geom',name='visual_super_'+name,type='mesh',mesh='metric_'+name,
            rgba=colors.get(name,'.5 .5 .5 1'),mass='0',contype='0',conaffinity='0',group='2')
    proxies=[]
    def box(name,lo,hi,source_parts):
        lo=np.array(lo);hi=np.array(hi);size=np.maximum((hi-lo)/2,1e-5)
        S(u,'geom',name='collision_'+name,type='box',pos=fmt((lo+hi)/2),size=fmt(size),
            mass='0',group='3',rgba='.28 .48 .65 .16')
        proxies.append(dict(name=name,type='box',center=((lo+hi)/2).tolist(),half_size=size.tolist(),source_parts=source_parts))
    # The central equipment box retains the holes/voids conservatively. Carbon
    # frame uses its convex hull, with individual swept discs for propellers.
    center_names=[n for n in info['parts'] if n not in ('carbon','motor','prop','lander','Led_front','Led_back')]
    lo=np.min([info['parts'][n]['bounds_min'] for n in center_names],axis=0)
    hi=np.max([info['parts'][n]['bounds_max'] for n in center_names],axis=0)
    box('central_equipment',lo,hi,center_names)
    # The 'motor' material also contains central electronics; cover these with
    # a central box, separately from the four peripheral motor cylinders.
    vertices=[]
    for ln in (ROOT/info['parts']['motor']['file']).read_text().splitlines():
        if ln.startswith('v '):vertices.append([float(x) for x in ln.split()[1:4]])
    v=np.array(vertices);center=v[np.linalg.norm(v[:,:2],axis=1)<.085]
    if len(center):box('central_motor_material',center.min(0),center.max(0),['motor:central'])
    S(u,'geom',name='collision_carbon',type='mesh',mesh='metric_carbon',mass='0',group='3',rgba='.28 .48 .65 .14')
    proxies.append(dict(name='carbon',type='convex_mesh',source_parts=['carbon']))
    landing=json.loads((ROOT/'revision/data/custom_landing_geometry.json').read_text())
    for p in landing['parts']:
        name='landing_%03d'%p['index'];S(asset,'mesh',name=name,file=str(ROOT/p['file']))
        color='.29 .34 .37 1' if any(k in p['source_part'] for k in ['支腿','滑橇','缓冲']) else '.52 .59 .64 1'
        S(u,'geom',name='visual_'+name,type='mesh',mesh=name,rgba=color,mass='0',contype='0',conaffinity='0',group='2')
        c=p['collision'];S(u,'geom',name='collision_'+name,type='box',pos=fmt(c['center']),size=fmt(c['half_size']),
            quat=fmt(c['quat_wxyz']),mass='0',group='3',rgba='.28 .48 .65 .13')
        proxies.append(dict(name=name,source_parts=[p['source_part']],**c))
    # Proposed central adapter standoffs clear all four swept propeller disks.
    plate_top=landing['plate_top_z_m'];frame_bottom=info['parts']['carbon']['bounds_min'][2]
    for i,(x,y) in enumerate([(.025,.025),(-.025,.025),(-.025,-.025),(.025,-.025)]):
        S(u,'geom',name='adapter_standoff%d'%i,type='capsule',fromto=fmt([x,y,plate_top,x,y,frame_bottom]),
            size='.003',mass='0',group='1',rgba='.58 .62 .65 1')
    for name in ('Led_front','Led_back'):
        p=info['parts'][name];box(name,p['bounds_min'],p['bounds_max'],[name])
    for rotor in info['rotors']:
        i=rotor['index'];p=np.array(rotor['axis_center_m']);r=rotor['prop_radius_m']
        vz=rotor['prop_z_interval_m'];z=sum(vz)/2
        S(u,'geom',name='collision_rotor_sweep%d'%i,type='cylinder',pos=fmt([p[0],p[1],z]),
            size=fmt([r,(vz[1]-vz[0])/2]),mass='0',group='3',rgba='.34 .57 .67 .18')
        S(u,'geom',name='collision_motor%d'%i,type='cylinder',pos=fmt(p),size='.017 .018',
            mass='0',group='3',rgba='.28 .48 .65 .14')
        proxies.append(dict(name='rotor_sweep%d'%i,type='cylinder',center=[float(p[0]),float(p[1]),z],
            radius=r,half_height=(vz[1]-vz[0])/2,source_parts=['prop:quadrant%d'%i]))
    # Forward camera is explicitly part of the custom modification, separate
    # from upstream SUPER. Its bounding primitive remains provisional.
    S(u,'geom',name='collision_body_camera',type='box',pos='.10 0 .035',size='.014 .045 .014',
        mass='0',group='3',rgba='.28 .48 .65 .16')
    # Supplied D435 CAD is in camera-link coordinates: long housing axis Y.
    # Reuse it without scaling; retain separate conservative collision boxes.
    ref=ROOT/'mujoco_closed_loop/vendor/reference_install'
    d435=json.loads((ref/'d435_manifest.json').read_text())
    tool=u.find(".//body[@name='tool']")
    tool.find("geom[@name='d435']").set('rgba','0 0 0 0')
    for p in d435['parts']:
        name='metric_'+p['mesh'];S(asset,'mesh',name=name,file=str(ref/'d435'/(p['mesh']+'.stl')))
        S(u,'geom',name='visual_body_'+name,type='mesh',mesh=name,pos='.10 0 .035',
            rgba=p['rgba'],mass='0',contype='0',conaffinity='0',group='2')
        S(tool,'geom',name='visual_wrist_'+name,type='mesh',mesh=name,pos='.045 0 .065',xyaxes='0 0 1 0 1 0',
            rgba=p['rgba'],mass='0',contype='0',conaffinity='0',group='2')
    cam=u.find("camera[@name='navigation']");cam.set('pos','.115 0 .035')
    cam=tool.find("camera[@name='wrist_rgbd']");cam.set('pos','.045 0 .079');cam.set('xyaxes','0 -1 0 -1 0 0')
    # Separate SDK visual diameter (50 mm) from the existing 68 mm collision
    # proxy; appearance is a discrete approximation, not a native continuum CAD.
    for i in range(segments):
        body=u.find(".//body[@name='seg%d']"%i);g=body.find("geom[@name='arm%d']"%i)
        g.set('rgba','.28 .48 .65 .13');g.set('group','3')
        S(body,'geom',name='arm_visual%d'%i,type='capsule',fromto=g.get('fromto'),size='.025',
            mass='0',contype='0',conaffinity='0',group='2',rgba='.51 .58 .60 1')
    mount=u.find("body[@name='mount']")
    for g in mount.findall('geom'):
        if g.get('type')=='cylinder':g.set('group','3');g.set('rgba','.28 .48 .65 .13')
    floor=world.find("geom[@name='floor']");floor.set('rgba','.96 .965 .97 1')
    info['runtime_geometry']=dict(proxies=proxies,mount_top_m=[0,0,-.03],
        mount_status='candidate central adapter; mounting holes and load capacity require verification',
        landing_gear='native long landing assembly; upstream short feet removed',
        landing_bounds_m=landing['bounds_m'],
        rigid_z_interval_m=[landing['bounds_m'][0][2],info['bounds_m'][1][2]],
        body_envelope_center_z_m=(landing['bounds_m'][0][2]+info['bounds_m'][1][2])/2,
        body_envelope_height_m=info['bounds_m'][1][2]-landing['bounds_m'][0][2],
        flexible_visual_diameter_m=.05,flexible_collision_diameter_m=.068,
        camera_status='provisional custom body camera bounds',
        rotated_propeller_sweep_included=True,mass_identified=False)
    return tostring(root,encoding='unicode'),info

def allocation_matrix(model):
    """Derive the actual thrust-to-wrench map from this compiled model."""
    cols=[]
    for i in range(4):
        a=model.actuator('thrust%d'%i).id;s=model.site('rotor%d'%i).id
        p=model.site_pos[s];g=model.actuator_gear[a]
        if not np.allclose(g[:3],[0,0,1]):raise ValueError('Only body-parallel rotor axes supported')
        torque=np.cross(p,g[:3])+g[3:]
        cols.append([g[2],*torque])
    a=np.array(cols).T
    if np.linalg.matrix_rank(a)!=4:raise ValueError('Singular rotor allocation')
    return a
