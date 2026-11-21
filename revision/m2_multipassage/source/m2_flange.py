"""Inspection flange, datum and bracket for the M2 industrial cell, in ENU metres."""
from functools import lru_cache

import numpy as np

from m0_scene import Piece, brick, turning, facets, basis, inspection_scene
from m0_scene import C as OLD_FACE, B as OLD_FRAME
from m2_scene import FACE, AXIS, STANDOFF, pipes

FRAME = basis(AXIS)
OLD_SCALE = np.array([.72, .72, .714])
BOLT_COUNT = 8


def _socket_face():
    """Flat annulus machined through to a true hexagonal recess."""
    count = 48
    a = 2*np.pi*np.arange(count)/count
    # Six planar flats meet the original socket's six corner positions.
    sector = np.mod(a, np.pi/3.)-np.pi/6.
    inner = .0039*np.cos(np.pi/6.)/np.cos(sector)
    periphery = np.column_stack([np.cos(a),np.sin(a)])
    rings = []
    for radius, depth in [(inner,-.00735),(inner,-.01070),(.00830,-.01070)]:
        xy = np.broadcast_to(radius,(count,))[:,None]*periphery
        rings.append(np.column_stack((xy,np.full(count,depth))))
    faces=[]
    for layer in range(2):
        for j in range(count):
            i=layer*count+j;k=layer*count+(j+1)%count
            faces.extend(((i,k,k+count),(i,k+count,i+count)))
    return facets(np.vstack(rings),np.asarray(faces))


def _hardware():
    body, bracing = [next((a,b,r) for n,a,b,r,_ in pipes() if n==key)
                     for key in ('inspection_flange','flange_support')]
    result={'inspection_flange':body,'flange_support':bracing}
    u=np.cross(AXIS,[0.,0.,1.]);u/=np.linalg.norm(u)
    v=np.cross(AXIS,u)
    for j in range(BOLT_COUNT):
        angle=2*np.pi*j/BOLT_COUNT
        center=FACE+.083*(np.cos(angle)*u+np.sin(angle)*v)
        result['flange_bolt_%d'%j]=(center-.011*AXIS,center+.002*AXIS,.010)
    return result


def _check(mesh, cylinder):
    a,b,r=cylinder;direction=np.asarray(b)-a;length=np.linalg.norm(direction);unit=direction/length
    local=mesh.v-a;height=local@unit;radial=np.linalg.norm(local-height[:,None]*unit,axis=1)
    clearance=min(height.min(),length-height.max(),r-radial.max())
    if clearance < -3e-8:
        raise ValueError('%s outside registered collision cover by %.3f mm' %(mesh.name,-1000*clearance))
    return float(clearance)


@lru_cache(maxsize=1)
def _assembly():
    parts=[];audit=[];covers=_hardware()
    def store(piece,cover):
        clearance=_check(piece,covers[cover])
        audit.append(dict(name=piece.name,cover='COL_'+cover,vertex_min_slack_m=clearance))
        parts.append(piece)
    def add(mesh,center,axes,color,group,name,cover,metal=.52,rough=.40):
        vertices,normals,triangles=mesh
        store(Piece(vertices@axes.T+center,normals@axes.T,triangles,color,metal,rough,1.,0.,group,name),cover)
    def ring(profile,center,color,group,name,cover,steps=96,metal=.52):
        add(turning(profile,steps),center,FRAME,color,group,name,cover,metal)
    # Retain the complete locked die-turned rim, sleeve, rear instrument,
    # three deep engravings, rear barrel and 36 engraved angular graduations.
    # The eight original sockets are re-machined onto the registered M2 bolts.
    for part in inspection_scene()[0]:
        if not part.group.startswith('flange') or part.name in ('socket housing','recessed hex'):
            continue
        local=(part.v-OLD_FACE)@OLD_FRAME
        normals=part.n@OLD_FRAME/OLD_SCALE
        normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-15)
        store(Piece((local*OLD_SCALE)@FRAME.T+FACE,normals@FRAME.T,part.f,
                    part.color,part.metal,part.rough,part.opacity,part.grid,part.group,part.name),
              'inspection_flange')

    # The camera measures the centre of the engraved surface at the surveyed FACE.
    ring([(0,.00035),(.061,.00035),(.063,.0023),(.063,.009),(0,.009)],FACE,
         '#759091','flange_face','surveyed face','inspection_flange')
    for radius,width,color,label in [(.035,.0012,'#c3915f','copper target ring'),
                                     (.052,.0007,'#abc1bc','silver gauge ring'),
                                     (.006,.0010,'#e0bc83','copper centre ring')]:
        ring([(radius-width,.00007),(radius+width,.00007),
              (radius+width,.00042),(radius-width,.00042)],FACE,
             color,'flange_face',label,'inspection_flange',steps=104)
    ring([(0,.00014),(.00070,.00014),(.00075,.00035),(0,.00035)],FACE,
         '#536c72','flange_face','centre dimple','inspection_flange',steps=24)
    for j in range(BOLT_COUNT):
        key='flange_bolt_%d'%j
        start,_,_=covers[key];c=start+.011*AXIS
        ring([(.0082,-.01068),(.00910,-.01068),(.00992,-.00972),(.00992,.00182),
              (.00855,.00182),(.00818,-.00864)],c,
             '#6b7c7c','flange_fasteners','counterbored shoulder %02d'%j,key,steps=48,metal=.65)
        head_axes=FRAME@np.array([[np.cos(j*.31),-np.sin(j*.31),0],
                                  [np.sin(j*.31),np.cos(j*.31),0],[0,0,1.]])
        add(_socket_face(),c,head_axes,'#687878','flange_fasteners','recessed hex wall %02d'%j,key,metal=.61)
        add(turning([(0,-.00733),(.00392,-.00733),(.00392,-.0068),(0,-.0068)],6),
             c,head_axes,'#34494e','flange_fasteners','hex socket floor %02d'%j,key,metal=.30)
    # Box-section cantilever, triangular shadow-line braces and machined
    # mounting collars all fit the exact finite registered support cylinder.
    a,b,r=covers['flange_support'];direction=b-a;length=np.linalg.norm(direction);basis_s=basis(direction)
    side,up,ax=basis_s.T
    def bar(name,start,end,width,depth,color):
        span=end-start
        add(brick([width,depth,np.linalg.norm(span)],.0022),(start+end)/2,basis(span),
            color,'flange_bracket',name,'flange_support',metal=.58)
    bar('central cantilever',a+.006*ax,b-.008*ax,.040,.044,'#718489')
    for sign in (-1.,1.):
        bar('diagonal gusset',a+.025*ax+sign*.013*side-.008*up,
            b-.022*ax+sign*.022*side+.013*up,.010,.013,'#859894')
    for z,band,col in [(.008,.010,'#849b99'),(length-.016,.012,'#60757a')]:
        profile=[(0,0),(.035,0),(.038,.0018),(.038,band-.0015),(.036,band),(0,band)]
        add(turning(profile,48),a+z*ax,basis_s,col,'flange_bracket','anchor collar','flange_support')
    return parts,audit


def flange_parts():
    """M0/M1-compatible component list for the single shared mesh and UV baker."""
    return list(_assembly()[0])


def flange_metadata():
    parts,audit=_assembly()
    return dict(face_enu_m=FACE.tolist(),axis_enu=AXIS.tolist(),optical_distance_m=STANDOFF,
                socket_count=BOLT_COUNT,graduation_count=sum(x.name=='engraved scale' for x in parts),
                source_geometry='M0 inspection_scene turned shell and M1 surveyed-face design',
                pieces=len(parts),triangles=sum(len(x.f) for x in parts),
                all_inside_registered_collision_solids=True,part_covers=audit)


def camera_audit(plant, output, pose=None):
    """Render the registered rigid wrist and observer at the surveyed inspection pose."""
    import hashlib
    import json
    from pathlib import Path
    import mujoco
    from PIL import Image
    from m2_plant import option,place,task_pose
    from m2_inspection_ik import geom_ids
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    m=plant.m;d=mujoco.MjData(m);datum=task_pose(plant)
    task=datum if pose is None else pose
    place(m,d,task['body'],task['theta_deg'],task['phi_deg'],task['yaw_deg'])
    camera=m.camera('wrist_rgbd').id;R=d.cam_xmat[camera].reshape(3,3)
    width,height=480,360;fx=height*.5/np.tan(np.deg2rad(m.cam_fovy[camera])/2)
    def pixel(point):
        x,y,z=R.T@(np.asarray(point)-d.cam_xpos[camera]);depth=-z
        return [float((width-1)/2+fx*x/depth),float((height-1)/2-fx*y/depth),float(depth)]
    wrist= mujoco.Renderer(m,height,width)
    observer=mujoco.Renderer(m,540,960)
    try:
        wrist.update_scene(d,camera='wrist_rgbd',scene_option=option())
        colour=wrist.render().copy()
        wrist.enable_depth_rendering();depth=wrist.render().copy();wrist.disable_depth_rendering()
        ids=geom_ids(wrist)
        observer.update_scene(d,camera='observer_dock',scene_option=option());third=observer.render().copy()
    finally:
        wrist.close();observer.close()
    Image.fromarray(colour).save(output/'flange-wrist.png')
    Image.fromarray(np.uint16(np.rint(1000*np.clip(depth,0,65.535)))).save(output/'flange-depth-mm.png')
    Image.fromarray(third).save(output/'flange-oblique.png')
    def sample(at):
        px,py,z=pixel(at);i,j=int(round(px)),int(round(py))
        if not (1<=i<width-1 and 1<=j<height-1):
            raise ValueError('Surveyed landmark lies outside the wrist raster')
        geom=int(ids[j,i,0]);kind=int(ids[j,i,1])
        name=m.geom(geom).name if geom>=0 and kind==int(mujoco.mjtObj.mjOBJ_GEOM) else None
        return dict(uv=[px,py],geom=name,expected_depth_m=z,
                    actual_depth_m=float(depth[j,i]),rgb=colour[j,i,:].tolist())
    covers=_hardware();bolts=[]
    for i in range(8):
        start,_,_=covers['flange_bolt_%d'%i]
        bolts.append(sample(start+.00366*AXIS))
    center=sample(FACE+.00014*AXIS)
    is_hardware=lambda item: 'flange' in (item['geom'] or '').lower()
    n=sum(is_hardware(x) and abs(x['actual_depth_m']-x['expected_depth_m'])<=.002 for x in bolts)
    metal_ids=[i for i in range(m.ngeom) if is_hardware(dict(geom=m.geom(i).name)) and m.geom_group[i]==0]
    occupancy=np.isin(ids[:,:,0],metal_ids)&(ids[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
    r,g,b=colour.astype(float).transpose(2,0,1)
    copper=occupancy&(r-g>13)&(g-b>10)
    result=dict(scene=plant.scene_digest,qpos_sha256=hashlib.sha256(d.qpos.tobytes()).hexdigest(),
        model_sha256=hashlib.sha256(plant.xml.encode()).hexdigest(),source='static FK, native MuJoCo camera render',
        camera='wrist_rgbd',pinhole=dict(width=width,height=height,fx_px=float(fx),cx_px=239.5,cy_px=179.5),
        center=center,bolts=bolts,socket_centers_depth_verified=n,flange_pixels=int(occupancy.sum()),
        copper_pixels=int(copper.sum()),raw_rgb_sha256=hashlib.sha256(colour.tobytes()).hexdigest(),
        stand_off_to_datum_m=pixel(FACE)[2],observer='observer_dock',fit=flange_metadata(),
        pose={k:task[k] for k in ('body','theta_deg','phi_deg','yaw_deg')},
        true_tcp_error_m=float(np.linalg.norm(d.site('tip').xpos-np.asarray(datum['tool']))),
        optical_datum_offset_px=float(np.linalg.norm(np.asarray(pixel(FACE)[:2])-[239.5,179.5])))
    (output/'flange-pixel-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    if not (.12<=result['stand_off_to_datum_m']<=.18 and is_hardware(center) and
            abs(center['actual_depth_m']-center['expected_depth_m'])<=.002 and n>=7 and copper.sum()>50):
        raise ValueError('Optical flange inspection failed; see flange-pixel-audit.json')
    return result


if __name__=='__main__':
    import argparse
    from pathlib import Path
    from m2_plant import build,portable,SCENE
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene',type=Path,default=SCENE)
    parser.add_argument('--portable',action='store_true')
    parser.add_argument('--audit-output',required=True,type=Path)
    args=parser.parse_args()
    report=camera_audit((portable if args.portable else build)(args.scene),args.audit_output)
    print('Wrist depth at centre (mm):',1000*report['center']['actual_depth_m'],
          'verified socket pixels:',report['socket_centers_depth_verified'],'of 8')
