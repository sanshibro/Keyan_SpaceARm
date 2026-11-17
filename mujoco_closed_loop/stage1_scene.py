"""Stage-1 visual assembly and task-space layout; no autonomy is implied."""
from pathlib import Path
from xml.etree.ElementTree import fromstring,SubElement as S,tostring
import json
from model import model_xml
from qingyu_model import ARC_LENGTH,VENDOR
import yaml

HERE=Path(__file__).resolve().parent
CAD=HERE/'vendor/super_converted'
FULL_CAD=CAD/'full_assembly'
REFERENCE_CAD=HERE/'vendor/reference_install'
PASSAGES=(('gap_1',-1.40,-1.00,-.28,.96,1.55),
          ('gap_2',-.05,.05,.73,1.00,1.56),
          ('gap_3',1.35,-.78,-.08,.96,1.53))

def add_geom(parent,name,kind,**kw):
    return S(parent,'geom',name=name,type=kind,**{k:str(v) for k,v in kw.items()})

def build_scene():
    root=fromstring(model_xml(fixed=True,camera_profile='grasp_visible',nav_camera_profile='front'))
    root.set('model','SUPER_Qingyu_stage1_visual_review')
    assets=root.find('asset')
    for name in ('main_board','second_board','battery_board','foot'):
        S(assets,'mesh',name='super_'+name,file=str(CAD/(name+'.obj')))
    for path in sorted(FULL_CAD.glob('*.obj')):
        S(assets,'mesh',name='super_full_'+path.stem,file=str(path))
    d435_parts=json.loads((REFERENCE_CAD/'d435_manifest.json').read_text())['parts']
    for part in d435_parts:
        S(assets,'mesh',name='ref_'+part['mesh'],
          file=str(REFERENCE_CAD/'d435'/(part['mesh']+'.stl')))
    S(assets,'texture',name='daylight_sky',type='skybox',builtin='gradient',width='512',height='512',rgb1='.76 .86 .94',rgb2='.97 .98 1')
    S(assets,'texture',name='floor_grid',type='2d',builtin='checker',width='1024',height='1024',rgb1='.49 .54 .57',rgb2='.54 .59 .61')
    S(assets,'material',name='floor_mat',texture='floor_grid',texrepeat='8 8',reflectance='.015')
    S(assets,'material',name='carbon',rgba='.075 .105 .13 1',reflectance='.22')
    S(assets,'material',name='accent',rgba='.05 .64 .75 1',reflectance='.24')
    visual=root.find('visual');S(visual,'global',offwidth='1440',offheight='810');S(visual,'rgba',haze='.76 .86 .94 1');S(visual,'headlight',ambient='.45 .45 .45',diffuse='.50 .50 .50')
    world=root.find('worldbody')
    for old in list(world.findall('light'))+list(world.findall('geom')):world.remove(old)
    S(world,'light',pos='-2 -3 5',dir='.3 .4 -1',diffuse='1 .96 .88',specular='.35 .35 .35',castshadow='true')
    S(world,'light',pos='2 2 4',dir='-.3 -.3 -1',diffuse='.55 .72 1',specular='.15 .2 .3')
    add_geom(world,'studio_floor','plane',size='7 7 .1',material='floor_mat',friction='1 .01 .001')
    # A restrained floor route gives the start, alternating gates, and pickup
    # target a readable order from the primary overview camera.
    route=[(-.64,-2.57),(-.64,-1.40),(.25,-.70),(.39,-.05),
           (-.25,.65),(-.43,1.35),(.65,2.35)]
    for i,((xa,ya),(xb,yb)) in enumerate(zip(route,route[1:])):
        add_geom(world,f'route_segment_{i}','capsule',
                 fromto=f'{xa} {ya} .009 {xb} {yb} .009',size='.009',
                 rgba='.94 .72 .34 .72',contype='0',conaffinity='0')
    for i,(x,y) in enumerate((route[0],route[1],route[3],route[5],route[-1])):
        add_geom(world,f'route_node_{i}','cylinder',pos=f'{x} {y} .006',
                 size='.045 .004',rgba='.98 .79 .43 .82',
                 contype='0',conaffinity='0')
    for name,pos,size,rgba in [
        ('left_wall','-2.35 0 1.35','.05 3.25 1.35','.47 .53 .55 .12'),
        ('right_wall','2.35 0 1.35','.05 3.25 1.35','.47 .53 .55 .12'),
        ('back_wall','0 3.25 1.35','2.4 .05 1.35','.62 .68 .67 .12')]:
        add_geom(world,name,'box',pos=pos,size=size,rgba=rgba)
    # Three *physical* narrow apertures across the only corridor. Their
    # alternating horizontal offsets require a whole-body zigzag. The low
    # sill/ceiling makes arm retraction part of the eventual flight task.
    for name,y,xlo,xhi,zlo,zhi in PASSAGES:
        wall_color='.22 .30 .34 .10'
        for suffix,xa,xb,za,zb in [
            ('left',-2.30,xlo,0,2.55),('right',xhi,2.30,0,2.55),
            ('sill',xlo,xhi,0,zlo),('lintel',xlo,xhi,zhi,2.55)]:
            add_geom(world,name+'_'+suffix,'box',pos=f'{(xa+xb)/2} {y} {(za+zb)/2}',size=f'{(xb-xa)/2} .055 {(zb-za)/2}',rgba=wall_color)
        for suffix,x in [('l',xlo),('r',xhi)]:
            add_geom(world,name+'_edge_'+suffix,'box',pos=f'{x} {y-.057} {(zlo+zhi)/2}',size=f'.012 .007 {(zhi-zlo)/2}',rgba='.05 .65 .70 1',contype='0',conaffinity='0')
        for suffix,z in [('bottom',zlo),('top',zhi)]:
            add_geom(world,name+'_edge_'+suffix,'box',pos=f'{(xlo+xhi)/2} {y-.057} {z}',size=f'{(xhi-xlo)/2} .007 .012',rgba='.05 .65 .70 1',contype='0',conaffinity='0')
        frame_color='.08 .17 .20 1'
        for suffix,x in [('left',xlo-.018),('right',xhi+.018)]:
            add_geom(world,name+'_portal_'+suffix,'box',
                     pos=f'{x} {y-.07} {(zlo+zhi)/2}',
                     size=f'.018 .018 {(zhi-zlo)/2}',rgba=frame_color,
                     contype='0',conaffinity='0')
        for suffix,z in [('lower',zlo-.018),('upper',zhi+.018)]:
            add_geom(world,name+'_portal_'+suffix,'box',
                     pos=f'{(xlo+xhi)/2} {y-.07} {z}',
                     size=f'{(xhi-xlo)/2} .018 .018',rgba=frame_color,
                     contype='0',conaffinity='0')
        add_geom(world,name+'_threshold','box',pos=f'0 {y} .025',
                 size='2.3 .075 .025',rgba='.12 .19 .22 1',
                 contype='0',conaffinity='0')
    # Offset columns create a slalom between the gates. The bright, stacked
    # placeholder is replaced with an industrial observation tower.
    for name,x,y,r,h in [('obstacle_a',-.22,-.70,.15,1.10),
                         ('obstacle_b',.25,.65,.16,1.35),
                         ('obstacle_c',-1.18,1.92,.16,1.00)]:
        add_geom(world,name,'cylinder',pos=f'{x} {y} {h/2}',
                 size=f'{r} {h/2}',rgba='.24 .31 .35 1')
        add_geom(world,name+'_top','cylinder',pos=f'{x} {y} {h+.012}',
                 size=f'{r+.012} .012',rgba='.06 .59 .66 1',
                 contype='0',conaffinity='0')
    add_geom(world,'observation_tower','box',pos='1.42 .60 .82',
             size='.14 .14 .82',rgba='.16 .22 .26 1')
    for i,z in enumerate((.35,.95,1.50)):
        add_geom(world,f'observation_tower_band_{i}','box',
                 pos=f'1.42 .60 {z}',size='.145 .145 .018',
                 rgba=('.06 .68 .74 1' if i!=1 else '.93 .54 .22 1'),
                 contype='0',conaffinity='0')
    add_geom(world,'offset_obstacle','cylinder',pos='-1.50 .35 .56',size='.18 .56',rgba='.32 .39 .42 1')
    add_geom(world,'target_pedestal','cylinder',pos='.65 2.35 .48',size='.23 .48',rgba='.28 .33 .38 1')
    add_geom(world,'target_object','sphere',pos='.65 2.35 1.02',size='.055',rgba='.95 .17 .15 1')
    add_geom(world,'basket_base','box',pos='-1.52 -2.48 .055',size='.32 .32 .055',rgba='.16 .32 .4 1')
    for i,(x,y,sx,sy) in enumerate([(-1.52,-2.77,.33,.035),(-1.52,-2.19,.33,.035),(-1.81,-2.48,.035,.29),(-1.23,-2.48,.035,.29)]):
        add_geom(world,'basket_wall%d'%i,'box',pos=f'{x} {y} .23',size=f'{sx} {sy} .18',rgba='.08 .48 .58 .85')
    # Sparse architectural details retain clear, readable task geometry.
    for i in range(6):
        add_geom(world,'back_pilaster%d'%i,'box',pos=f'{-2.2+i*.9:.2f} 3.18 1.35',size='.035 .025 1.35',rgba='.34 .41 .44 1')
    u=world.find("body[@name='uav']");u.set('pos','-.64 -2.57 1.28')
    # The archived SUPER simulator mesh is the visual authority for the
    # complete airframe. Keep provisional primitives for mass only.
    colors={'Lidar':'.83 .87 .89 1','metal_lidar':'.21 .25 .28 1',
            'carbon':'.09 .11 .14 1','motor':'.19 .20 .22 1',
            'prop':'.085 .095 .11 1','metal':'.27 .30 .32 1',
            'lander':'.12 .13 .15 1','PCB':'.13 .29 .22 1',
            'yellow':'.9 .7 .10 1','green':'.17 .65 .23 1',
            'Led_front':'.12 .75 .85 1','Led_back':'.88 .18 .14 1',
            'mars_logo':'.9 .9 .9 1','usb':'.65 .68 .70 1'}
    for path in sorted(FULL_CAD.glob('*.obj')):
        add_geom(u,'super_assembly_'+path.stem,'mesh',mesh='super_full_'+path.stem,
                 rgba=colors.get(path.stem,'.45 .45 .45 1'),mass='0',contype='0',conaffinity='0',group='2')
    fuselage=u.find("geom[@name='fuselage']");fuselage.set('rgba','.12 .16 .19 1');fuselage.set('mass','.647');fuselage.set('size','.06 .055 .03');fuselage.set('group','3')
    q='.70710678 -.70710678 0 0'
    for name,z,mass in [('main_board',.055,.115),('second_board',-.055,.040),('battery_board',.105,.011)]:
        add_geom(u,'super_'+name+'_visual','mesh',mesh='super_'+name,pos=f'0 0 {z}',quat=q,material='carbon',contype='0',conaffinity='0',mass=mass,group='3')
    for site in u.findall('site'):
        if site.get('name','').startswith('rotor'):site.set('rgba','0 0 0 0')
    for i,(x,y) in enumerate(((.099,.099),(-.099,.099),(-.099,-.099),(.099,-.099))):
        add_geom(u,'motor%d'%i,'cylinder',pos=f'{x} {y} .08',size='.024 .017',rgba='.22 .27 .29 1',mass='.045',contype='0',conaffinity='0',group='3')
        add_geom(u,'prop%d'%i,'box',pos=f'{x} {y} .103',size='.13 .012 .003',rgba='.06 .08 .1 1',mass='.003',contype='0',conaffinity='0',group='3')
    add_geom(u,'lidar_mid360','cylinder',pos='0 0 .16',size='.041 .037',rgba='.85 .88 .85 1',mass='.16',group='3')
    add_geom(u,'nuc','box',pos='0 0 .012',size='.055 .047 .025',rgba='.15 .17 .19 1',mass='.12',contype='0',conaffinity='0',group='3')
    add_geom(u,'battery','box',pos='0 0 .14',size='.05 .027 .015',rgba='.18 .24 .35 1',mass='.145',contype='0',conaffinity='0',group='3')
    for camera in list(u.findall('camera')):u.remove(camera)
    # The reference assembly places its forward D435 against the central
    # airframe, below the rotor plane. Adapt that compact mounting to SUPER.
    add_geom(u,'body_camera_bridge','box',pos='0 .074 .036',size='.018 .028 .003',
             rgba='.10 .15 .18 1',mass='0',contype='0',conaffinity='0')
    add_geom(u,'body_camera_upright','box',pos='0 .100 .025',size='.018 .003 .013',
             rgba='.10 .15 .18 1',mass='0',contype='0',conaffinity='0')
    body_cam=S(u,'body',name='body_d435_mount',pos='0 .105 .015',
               xyaxes='0 0 -1 1 0 0')
    S(body_cam,'inertial',pos='0 0 0',mass='.08',diaginertia='.000060 .000010 .000060')
    add_geom(body_cam,'body_d435_collision','box',size='.0125 .045 .0126',
             rgba='0 0 0 0',mass='0',group='3')
    for part in d435_parts:
        add_geom(body_cam,'body_'+part['mesh'],'mesh',mesh='ref_'+part['mesh'],
                 rgba=part['rgba'],mass='0',contype='0',conaffinity='0',group='2')
    S(body_cam,'camera',name='navigation',pos='0 0 -.013',
      xyaxes='0 -1 0 1 0 0',fovy='70')
    # The Qingyu RViz publisher, unlike the fixed-link Xacro, renders its
    # 0.28 m backbone as 15 black 0.05 m spheres at 0.02 m intervals.
    # Attach those exact markers to the nearest physical bending body.
    for i in range(8):
        seg=u.find(f".//body[@name='seg{i}']")
        physical=seg.find(f"geom[@name='arm{i}']")
        physical.set('group','3')
    sdk_visual=yaml.safe_load((VENDOR/'config/continuum_arm_visualize.yaml').read_text())
    count=int(sdk_visual['resolution']);radius=float(sdk_visual['marker_scale'])/2
    ds=ARC_LENGTH/8
    for j in range(count):
        arc=ARC_LENGTH*j/(count-1);i=min(7,int(arc/ds));local_z=arc-(i+.5)*ds
        seg=u.find(f".//body[@name='seg{i}']")
        add_geom(seg,f'qingyu_rviz_backbone_{j}','sphere',pos=f'0 0 {local_z:.6f}',
                 size=f'{radius}',rgba='.075 .075 .075 1',mass='0',contype='0',conaffinity='0',group='2')
    tool=u.find(".//body[@name='tool']")
    for geom in list(tool.findall('geom')):
        if geom.get('name') in ('bracket','camera_standoff','d435'):
            tool.remove(geom)
    tool.remove(tool.find("camera[@name='wrist_rgbd']"))
    # Keep the supplied D435 CAD and saddle. The shorter front-side plate
    # remains parallel to the body camera plate, at the original wrist
    # height and downward optical direction. Both housing axes run across X.
    wrist_mount=S(tool,'body',name='wrist_camera_mount',pos='0 .080 .010',
                  xyaxes='0 1 0 1 0 0')
    S(wrist_mount,'inertial',pos='0 0 0',mass='.08',diaginertia='.000060 .000010 .000060')
    add_geom(wrist_mount,'wrist_d435_collision','box',size='.0125 .045 .0126',
             rgba='0 0 0 0',mass='0',group='3')
    for part in d435_parts:
        add_geom(wrist_mount,'wrist_'+part['mesh'],'mesh',mesh='ref_'+part['mesh'],
                 rgba=part['rgba'],mass='0',contype='0',conaffinity='0',group='2')
    for name,pos,size in [('camera_saddle','-.0145 0 .002','.002 .032 .010'),
                          ('camera_saddle_foot','-.020 0 .012','.006 .014 .002'),
                          ('camera_saddle_stem','-.015 0 .006','.002 .014 .008')]:
        add_geom(wrist_mount,name,'box',pos=pos,size=size,
                 rgba='.05 .51 .58 1',mass='0',contype='0',conaffinity='0')
    for side in (-1,1):
        add_geom(wrist_mount,f'camera_saddle_bolt_{side}','cylinder',
                 pos=f'-.017 {side*.018} .002',quat='.7071067812 0 .7071067812 0',
                 size='.002 .001',rgba='.7 .75 .78 1',mass='0',contype='0',conaffinity='0')
    add_geom(tool,'wrist_saddle_link','box',pos='0 .030 -.002',
             size='.018 .030 .003',rgba='.09 .17 .19 1',mass='.02',
             contype='0',conaffinity='0')
    S(wrist_mount,'camera',name='wrist_rgbd',pos='0 0 -.013',
      xyaxes='0 -1 0 1 0 0',fovy='70')
    for i in range(3):
        finger=tool.find(f"body[@name='finger{i}']")
        for geom in finger.findall('geom'):
            if geom.get('mesh'):
                geom.set('rgba','.075 .09 .11 1')
            elif geom.get('name')==f'pad{i}':geom.set('group','3')
        pad=finger.find(f"geom[@name='pad{i}']")
        add_geom(finger,f'finger_color_{i}','box',pos=pad.get('pos'),size='.005 .005 .018',
                 euler=pad.get('euler'),rgba=('.90 .18 .21 1' if i==0 else '.08 .68 .73 1'),
                 mass='0',contype='0',conaffinity='0',group='2')
    for side in (-1,1):
        for x in (-.07,.07):
            add_geom(u,f'super_foot_{side}_{x}','mesh',mesh='super_foot',pos=f'{x} {side*.11} -.08',quat='.70710678 .70710678 0 0',material='carbon',mass='0',contype='0',conaffinity='0',group='3')
        add_geom(u,'landing_rail_'+str(side),'capsule',fromto=f'-.09 {side*.11} -.14 .09 {side*.11} -.14',size='.008',rgba='.1 .13 .15 1',mass='0',contype='0',conaffinity='0',group='3')
        for x in (-.07,.07):
            add_geom(u,f'landing_leg_{side}_{x}','capsule',fromto=f'{x} {side*.075} -.025 {x} {side*.11} -.14',size='.006',rgba='.12 .14 .16 1',mass='0',contype='0',conaffinity='0',group='3')
    return tostring(root,encoding='unicode')


def build_flight_base():
    """The approved Stage 1 robot assembly with a dynamic UAV and empty room."""
    root=fromstring(build_scene())
    root.set('model','SUPER_Qingyu_approved_flight_assembly')
    world=root.find('worldbody')
    for geom in list(world.findall('geom')):world.remove(geom)
    S(world,'geom',name='studio_floor',type='plane',size='7 7 .1',material='floor_mat',friction='1 .01 .001')
    u=world.find("body[@name='uav']")
    u.set('pos','0 0 1.5')
    # Preserve the flight model's 2.25 kg robot mass while retaining the
    # approved Stage 1 meshes and short camera saddle as the visible model.
    for geom in u.findall('geom'):geom.set('mass','0')
    fuselage=u.find("geom[@name='fuselage']")
    fuselage.set('mass','1.42')
    fuselage.set('size','.14 .10 .06')
    wrist=u.find(".//body[@name='wrist_camera_mount']/inertial")
    wrist.set('mass','.13')
    S(u,'freejoint',name='flight')
    actuators=root.find('actuator')
    for i in range(4):
        S(actuators,'general',name=f'thrust{i}',site=f'rotor{i}',
          gear=f'0 0 1 0 0 {0.02*(-1)**i}',ctrlrange='0 12')
    return root


def build_collision_reference():
    """Approved robot collision geometry without the heavy display-only CAD."""
    root=build_flight_base()
    for parent in root.iter():
        for geom in list(parent.findall('geom')):
            if geom.get('contype')=='0' and geom.get('conaffinity')=='0':
                parent.remove(geom)
    used={geom.get('mesh') for geom in root.iter('geom') if geom.get('mesh')}
    asset=root.find('asset')
    for mesh in list(asset.findall('mesh')):
        if mesh.get('name') not in used:asset.remove(mesh)
    return root
