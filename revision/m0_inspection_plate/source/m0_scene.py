"""Metric inspection cell, object-space cutaway and registered aircraft meshes."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import ConvexHull
import trimesh
from m0_data import unit, basis, quat_mat, protocol

P=protocol()
C=np.asarray(P["target"],float)
A=unit(P["target_axis_enu"])
B=basis(A)
TEAL="#4b8583"
STEEL="#788d95"
LIGHT="#c7cfd0"
DARK="#435a65"
AMBER="#c38b59"

def rgb(c):
    if not isinstance(c,str):return np.asarray(c)[:3]
    return np.array([int(c[k:k+2],16)/255 for k in (1,3,5)])

@dataclass
class Piece:
    v:np.ndarray
    n:np.ndarray
    f:np.ndarray
    color:object
    metal:float
    rough:float
    opacity:float
    grid:float
    group:str
    name:str
    body:int=0

def facets(v,f):
    tri=v[f]
    norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    norm/=np.maximum(np.linalg.norm(norm,axis=1)[:,None],1e-14)
    return tri.reshape(-1,3),np.repeat(norm,3,axis=0),np.arange(tri.size//3,dtype=np.int32).reshape(-1,3)

def brick(size,chamfer=0):
    h=np.asarray(size)/2
    if chamfer<=0:
        mesh=trimesh.creation.box(extents=size);return facets(np.asarray(mesh.vertices),np.asarray(mesh.faces))
    cut=min(chamfer,np.min(h)*.42)
    v=[]
    for i in (-1,1):
        for j in (-1,1):
            for k in (-1,1):
                for axis in np.eye(3):
                    v.append(np.array([i,j,k])*(h-cut+axis*cut))
    v=np.array(v);f=ConvexHull(v).simplices.copy()
    norms=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    flip=(norms*v[f].mean(1)).sum(1)<0
    f[flip]=f[flip][:,::-1]
    return facets(v,f)

def turning(profile,steps=80):
    phi=np.arange(steps)*2*np.pi/steps
    circle=np.c_[np.cos(phi),np.sin(phi),np.zeros(steps)]
    vs=[];ns=[];fs=[]
    for j in range(len(profile)):
        r,z=profile[j];s,w=profile[(j+1)%len(profile)]
        if s==r and w==z:continue
        v0=circle*r+[0,0,z];v1=circle*s+[0,0,w]
        n=(w-z)*circle+np.array([0,0,r-s])
        n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-14)
        o=len(vs)*steps
        vs.extend((v0,v1));ns.extend((n,n))
        for k in range(steps):
            l=(k+1)%steps
            fs.extend(((o+k,o+l,o+steps+k),(o+l,o+steps+l,o+steps+k)))
    return np.vstack(vs),np.vstack(ns),np.asarray(fs,dtype=np.int32)

def inspection_scene():
    items=[];cert={}
    ident=np.eye(3)
    def add(mesh,pos,frame,color,group,name,metal=.35,rough=.43,opacity=1,grid=0):
        v,n,f=mesh
        items.append(Piece(v@frame.T+pos,n@frame.T,f,color,metal,rough,opacity,grid,group,name))
    def box(name,lo,hi,color,group,bevel=.006,**kw):
        lo=np.array(lo);hi=np.array(hi)
        add(brick(hi-lo,bevel),(lo+hi)/2,ident,color,group,name,**kw)
    def bar(name,p,q,width,height,color,group,**kw):
        l=np.asarray(p,float);r=np.asarray(q,float);d=r-l
        add(brick([width,height,np.linalg.norm(d)],min(width,height)*.1),(l+r)/2,basis(d),color,group,name,**kw)
    def radius(name,profile,group,color,center=C,orient=B,steps=96,**kw):
        add(turning(profile,steps),np.asarray(center),orient,color,group,name,**kw)
    # Each cover encloses its assigned drawn solids. Dimensions are metres in ENU.
    def cover(name,lo,hi,frame=ident,origin=(0,0,0)):
        cert[name]=dict(kind="box",lo=list(lo),hi=list(hi),frame=frame.tolist(),origin=list(origin))
    def annular(name,rin,rout,zmin,zmax):
        cert[name]=dict(kind="annular",rin=rin,rout=rout,zmin=zmin,zmax=zmax,frame=B.tolist(),origin=C.tolist())
    cover("foundation",[-1.1,-1.22,-.085],[2.95,1.12,0.])
    box("foundation",[-1.1,-1.22,-.085],[2.95,1.12,0.],"#d2d7d8","foundation",grid=1,metal=.06)
    cover("approach_deck",[-.55,-.82,0],[.585,.82,.825])
    box("approach pedestal",[-.51,-.782,.002],[.581,.783,.794],"#c9cdcb","approach_deck",bevel=.023)
    box("lower approach",[-.55,-.82,.765],[.585,.82,.825],"#c2c9c7","approach_deck",grid=1)
    cover("service_deck",[.58,-.82,0],[2.57,.82,1.265])
    box("sectioned service plinth",[.607,-.783,.002],[2.545,.798,1.231],"#b4bdbd","service_deck",bevel=.021)
    for x in (.86,1.44,2.02):
        box("access cover",[x-.196,-.788,.89],[x+.196,-.782,1.159],"#c7cdca","service_deck",bevel=.005)
        for j in range(7):
            box("vent slot",[x-.133+.042*j,-.790,1.083],[x-.11+.042*j,-.787,1.092],STEEL,"service_deck",bevel=.001)
    box("inspection gallery",[.58,-.82,1.195],[2.57,.82,1.265],"#bcc5c3","service_deck",grid=1,bevel=.012)
    for x in (.642,1.48,2.505):
        for y in (-.735,.72):
            box("gallery riser",[x-.029,y-.03,.77],[x+.029,y+.03,1.20],STEEL,"service_deck",bevel=.004)
    box("section edge",[.59,-.817,1.204],[2.56,-.806,1.216],DARK,"service_deck",bevel=.001)
    # Back wall is cut away on the reader's side; skin and structural ribs are distinct.
    cover("sidewall",[-.54,.712,.77],[2.57,.82,2.66])
    for j,(lo,hi) in enumerate(((-.53,.17),(.18,.96),(1.04,1.74),(1.75,2.55))):
        top=(1.64,1.81,2.47,2.50)[j]
        box("sectioned rear skin "+str(j),[lo,.765,1.26],[hi,.789,top],
            "#c3cfcf","sidewall",opacity=.16,rough=.56)
        box("base kick plate",[lo,.75,1.268],[hi,.791,1.40],LIGHT,"sidewall",rough=.57)
        box("top fold",[lo,.753,top-.008],[hi,.793,top+.006],"#a3b1b3","sidewall",opacity=.45)
    for x,top in ((-.515,1.64),(.172,1.65),(1.0,2.47),(1.745,2.47),(2.53,2.50)):
        box("rear rib",[x-.008,.758,1.29],[x+.008,.795,top],"#aababc","sidewall",bevel=.003,opacity=.38)
    box("rolled rear rail",[-.50,.741,1.43],[2.53,.765,1.439],"#adbabb","sidewall",opacity=.38)
    for xlo,xhi,top in ((1.52,2.01,1.68),(2.02,2.55,2.19)):
        box("service backplate",[xlo,.754,1.33],[xhi,.778,top],"#bec6c1","sidewall",bevel=.014,metal=.30)
    for j in range(7):
        box("mounting bay vent",[2.322,.751,1.86+.026*j],[2.477,.756,1.868+.026*j],
            "#7f9396","sidewall",bevel=.002,metal=.31)
    # Four registered inner planes. No cladding projects beyond x=[.95,1.05].
    g=P["gate"];x=g["center_x"];x0=x-g["thickness"]/2;x1=x+g["thickness"]/2
    half=g["half_width"];low=g["sill_z"];high=g["lintel_z"];post=g["post_width_y"];dep=g["beam_depth_z"]
    cover("sill",[x0,-.825,0],[x1,.825,low])
    cover("lintel",[x0,-.825,high],[x1,.825,2.665])
    cover("jambminus",[x0,-.825,0],[x1,-half,2.665])
    cover("jambplus",[x0,half,0],[x1,.825,2.665])
    box("sill structural backing",[x0+.013,-half,low-dep],[x1-.013,half,low],"#839396","sill",bevel=.004,metal=.43)
    box("registered lintel",[x0,-half,high],[x1,half,high+dep],"#b0bcbc","lintel",bevel=0,opacity=.28)
    for sign,key in ((-1,"jambminus"),(1,"jambplus")):
        ya,yb=sorted([sign*half,sign*(half+post)])
        box("sectioned backing web",[x0+.013,ya,1.265],[x1-.013,yb,high+dep],
            "#aabfc0",key,bevel=.003,opacity=.13 if sign<0 else .37)
        ya,yb=sorted([sign*(half+post),sign*.82])
        box("folded bulkhead web",[x0+.014,ya,1.245],[x1-.014,yb,2.60],
            "#a0b4b5",key,opacity=.10 if sign<0 else .47)
        # Sectioned edge reveal, outside the registered opening.
        ya,yb=sorted([sign*(half+.003),sign*(half+.011)])
        box("metal jamb reveal",[x0+.001,ya,low],[x0+.009,yb,high],TEAL,key,bevel=.001,metal=.62)
        for z in (1.44,1.78,2.06,2.33,2.54):
            y=sign*(half+.065)
            radius("flush fastener",[(0,0),(.009,0),(.010,.002),(.009,.005),(0,.005)],
                   key,DARK,[x0+.001,y,z],basis([1,0,0]),steps=12,metal=.72)
    box("lower bulkhead skin",[x0+.012,-half,0],[x1-.012,half,low-dep],
        "#acbbb7","sill",rough=.53,metal=.31)
    box("upper header skin",[x0+.012,-.82,high+dep],[x1-.012,.82,2.60],
        "#abc0c3","lintel",opacity=.28)
    for z,key in ((low-.010,"sill"),(high+.004,"lintel")):
        box("anodised reveal",[x0+.001,-half+.004,z],[x0+.010,half-.004,z+.005],TEAL,key,bevel=.001)
    for sign in (-1,1):
        for j in range(4):
            y=sign*(.27+j*.024)
            box("index mark",[x0+.0003,y-.005,low-.041],[x0+.004,y+.005,low-.016],AMBER,"sill",bevel=.0005)
    # Folded gasket, bevel, reveal and flush countersinks are continuous on both
    # faces. The inner lip follows the measured rectangular opening exactly.
    def portal_layer(name,profile,color,metal,rough):
        def loop(d,ch,at):
            l,r=-half-d,half+d;bot,top=low-d,high+d
            return np.array([[at,y,z] for y,z in ((l+ch,bot),(r-ch,bot),(r,bot+ch),(r,top-ch),
                (r-ch,top),(l+ch,top),(l,top-ch),(l,bot+ch))])
        loops=[loop(d,ch,xx) for d,ch,xx in profile]
        groups={}
        for a,b in zip(loops,loops[1:]+loops[:1]):
            for j in range(8):
                k=(j+1)%8
                for v in (np.array([a[j],a[k],b[j]]),np.array([b[j],a[k],b[k]])):
                    if np.linalg.norm(np.cross(v[1]-v[0],v[2]-v[0]))<1e-10:continue
                    if v[:,2].max()<=low+1e-10:key="sill"
                    elif v[:,2].min()>=high-1e-10:key="lintel"
                    elif v[:,1].max()<=-half+1e-10:key="jambminus"
                    elif v[:,1].min()>=half-1e-10:key="jambplus"
                    else:raise ValueError(name)
                    groups.setdefault(key,[]).append(v)
        triangles=np.asarray([v for tris in groups.values() for v in tris])
        volume6=np.einsum('ij,ij->',triangles[:,0],np.cross(triangles[:,1],triangles[:,2]))
        for key,value in groups.items():
            v=np.vstack(value);f=np.arange(len(v)).reshape(-1,3)
            if volume6<0:f=f[:,::-1]
            add(facets(v,f),np.zeros(3),ident,color,key,name,metal=metal,rough=rough)
    portal_layer("tunnel liner",[(0,0,x0),(.009,.008,x0),(.009,.008,x1),(0,0,x1)],
                 "#5d8683",.59,.38)
    for face,sgn in ((x0,1),(x1,-1)):
        def xx(a):return face+sgn*a
        portal_layer("machined outer shoulder",[(.008,.008,xx(.010)),(.074,.034,xx(.010)),
              (.078,.038,xx(.015)),(.078,.038,xx(.026)),(.008,.008,xx(.026))],"#52696c",.62,.38)
        portal_layer("folded satin bezel",[(.011,.009,xx(.008)),(.015,.012,xx(.003)),
              (.061,.029,xx(.003)),(.069,.033,xx(.009)),(.064,.030,xx(.011)),
              (.011,.009,xx(.011))],"#a5a99f",.55,.42)
        portal_layer("teal datum line",[(.006,.005,xx(.0020)),(.010,.008,xx(.0020)),
              (.010,.008,xx(.007)),(.006,.005,xx(.007))],"#508986",.45,.43)
        # Repeating fasteners and paired witness marks express scale on either side.
        for y,z,key in ([(y,low-.038,"sill") for y in (-.30,-.14,.14,.30)]+
                        [(y,high+.038,"lintel") for y in (-.30,-.14,.14,.30)]+
                        [(ss*(half+.038),z,"jambplus" if ss>0 else "jambminus")
                          for ss in (-1,1) for z in (low+.10,(low+high)/2,high-.10)]):
            R=basis([sgn,0,0])
            radius("sunk bezel head",[(0,0),(.0056,0),(.0068,.0013),(.0068,.003),(0,.003)],
                   key,"#586969",[xx(.0003),y,z],R,steps=20,metal=.49)
            radius("hex drive",[(0,0),(.0027,0),(.0027,.0006),(0,.0006)],
                   key,"#283d43",[xx(.0001),y,z],R,steps=6,metal=.32)
        for ss in (-1,1):
            for j in range(3):
                y=ss*(.292+.019*j)
                box("sill witness bars",[min(xx(.0002),xx(.002)),y-.003,low-.053],
                    [max(xx(.0002),xx(.002)),y+.003,low-.027],AMBER,"sill",bevel=.0003)
        for j in range(29):
            z=low+.065+j*.020
            box("metric reveal markings",[min(xx(.0005),xx(.0013)),half+.017,z],
                [max(xx(.0005),xx(.0013)),half+(.030 if j%5==0 else .023),z+.0011],DARK,
                "jambplus",bevel=0,metal=.32)
        for z in (1.58,2.445):
            bar("folded gusset",[xx(.026),.515,z],[xx(.026),.744,z+.087],
                .018,.027,"#879b9c","jambplus",metal=.56)
        for y0,y1,z0,z1 in ((.47,.787,1.31,1.96),(.47,.787,1.971,2.53)):
            box("press-formed web cassette",[min(xx(.012),xx(.037)),y0,z0],
                [max(xx(.012),xx(.037)),y1,z1],"#aebbba","jambplus",bevel=.010,metal=.43)
        for j in range(6):
            box("web ventilation lance",[min(xx(.009),xx(.015)),.578,2.27+.023*j],
                [max(xx(.009),xx(.015)),.729,2.278+.023*j],"#637e80","jambplus",bevel=.002)
        for y0,y1 in ((-.344,-.025),(-.014,.34)):
            box("removable sill panel",[min(xx(.009),xx(.016)),y0,1.332],
                [max(xx(.009),xx(.016)),y1,1.522],"#b6bfba","sill",bevel=.008,metal=.35)
    # The datum is the open mouth. The actual marked metal target face is recessed.
    annular("flange_lip",.087,.151,0,.066)
    radius("rolled annular rim",[(.091,.0055),(.141,.0055),(.149,.011),(.149,.026),(.136,.034),
                                 (.136,.052),(.096,.052),(.0875,.035),(.0875,.008)],
           "flange_lip","#9ca8a7",metal=.66,rough=.40)
    radius("bronze insert",[(.088,.004),(.095,.004),(.097,.007),(.095,.011),(.088,.011)],
           "flange_lip","#b7895a",metal=.71,rough=.32)
    radius("face index ring",[(.107,.0047),(.109,.0047),(.109,.007),(.107,.007)],
           "flange_lip","#528484",metal=.48)
    for j in range(8):
        a=j*np.pi/4+np.pi/8
        c=C+B@np.array([.124*np.cos(a),.124*np.sin(a),.0002])
        radius("socket housing",[(0,.0004),(.010,.0004),(.012,.002),(.012,.007),(0,.007)],
               "flange_lip","#647778",c,B,steps=28,metal=.60,rough=.40)
        radius("recessed hex",[(0,0),(.0055,0),(.0055,.002),(0,.002)],
               "flange_lip","#2e4148",c,B,steps=6,metal=.29,rough=.50)
    for j in range(36):
        a=j*np.pi/18
        u=np.array([np.cos(a),np.sin(a),0]);t=np.array([-np.sin(a),np.cos(a),0])
        coord=C+B@(.101*u+[0,0,.0049])
        R=B@np.column_stack([u,t,[0,0,1]])
        add(brick([.007 if j%3==0 else .004,.0011,.0007]),coord,R,
            "#566f70","flange_lip","engraved scale",metal=.28,rough=.51)
    annular("flange_tube",.080,.14,.049,.168)
    radius("bored inspection sleeve",[(.084,.05),(.132,.05),(.136,.055),(.113,.076),
                                     (.113,.147),(.127,.158),(.125,.166),(.081,.166),(.081,.059)],
           "flange_tube","#647b83",metal=.66,rough=.34)
    annular("flange_back",0,.143,.145,.324)
    radius("recessed instrument face",[(0,.151),(.089,.151),(.095,.158),(.095,.172),(0,.172)],
           "flange_back","#6d8588",metal=.52,rough=.38)
    for rr,col in ((.035,"#cf9a65"),(.007,"#cad9d5"),(.063,"#b6c5c1")):
        radius("instrument face engraving",[(rr-.0011,.1497),(rr+.0011,.1497),(rr+.0011,.152),(rr-.0011,.152)],
               "flange_back",col,metal=.4)
    radius("flange rear barrel",[(0,.167),(.116,.167),(.116,.177),(.091,.196),(.085,.292),
                                  (.092,.301),(.088,.312),(0,.312)],"flange_back",STEEL,metal=.68)
    end=C+.275*A; anchor=np.array([2.256,.734,1.435])
    cover("support",[2.08,.335,1.28],[2.38,.775,1.57])
    bar("cantilever",end,anchor,.050,.059,STEEL,"support",metal=.68)
    bar("diagonal gusset",end+.012*A,anchor+[.062,0,-.113],.028,.033,STEEL,"support")
    box("sidewall mounting shoe",[2.14,.717,1.297],[2.376,.754,1.553],"#657b7d","support",bevel=.011)
    for x in (2.162,2.351):
        for z in (1.324,1.529):
            radius("anchor bolt",[(0,0),(.011,0),(.012,.005),(.014,.009),(0,.009)],
                   "support","#a0a9a6",[x,.713,z],basis([0,1,0]),steps=6,metal=.66)
    # Faint metric floor marks are shaded in real world coordinates. The optical
    # window and the 35-mm tolerance locus are annotations, not additional solids.
    return items,cert

def robot_parts(m):
    pieces=[]
    smooth_sphere=trimesh.creation.icosphere(subdivisions=3)
    for i in range(m.ngeom):
        group,body,kind=int(m.geom_group[i]),int(m.geom_bodyid[i]),int(m.geom_type[i])
        rgba=m.geom_rgba[i]
        if body==0 or group not in (1,2) or rgba[3]<.05:continue
        name=m.geom(i).name or ("body_part_"+str(i))
        if kind==7:
            mid=int(m.geom_dataid[i]);va=m.mesh_vertadr[mid];fa=m.mesh_faceadr[mid]
            vi=m.mesh_vert[va:va+m.mesh_vertnum[mid]].astype(float)
            fi=m.mesh_face[fa:fa+m.mesh_facenum[mid]].copy()
            na=m.mesh_normaladr[mid]
            ni=m.mesh_normal[na:na+m.mesh_normalnum[mid]].astype(float)
            fn=m.mesh_facenormal[fa:fa+m.mesh_facenum[mid]]
            if np.array_equal(fn,fi):
                v,n,f=vi,ni,fi
            else:
                pairs=np.stack((fi.ravel(),fn.ravel()),axis=-1)
                unique,inv=np.unique(pairs,axis=0,return_inverse=True)
                v,n,f=vi[unique[:,0]],ni[unique[:,1]],inv.reshape(-1,3)
        elif kind in (2,4):
            scale=m.geom_size[i].copy()
            if kind==2:scale[:]=scale[0]
            v=np.asarray(smooth_sphere.vertices)*scale
            n=np.array(smooth_sphere.vertex_normals)/scale
            n/=np.linalg.norm(n,axis=1)[:,None];f=np.asarray(smooth_sphere.faces)
        elif kind in (3,5):
            r,l=m.geom_size[i,:2]
            mesh=trimesh.creation.capsule(height=2*l,radius=r,count=[12,20]) if kind==3 else trimesh.creation.cylinder(radius=r,height=2*l,sections=24)
            v,n,f=np.asarray(mesh.vertices),np.asarray(mesh.vertex_normals),np.asarray(mesh.faces)
        elif kind==6:v,n,f=brick(2*m.geom_size[i],.0005)
        else:raise ValueError((name,kind))
        R=quat_mat(m.geom_quat[i])
        color=rgba[:3].copy();metal=.55;rough=.43
        if "backbone" in name:color=rgb("#424f56");metal=.27;rough=.55
        if "finger" in name:color=rgb("#45828c");metal=.54
        if "prop" in name:color=rgb("#364650");metal=.28
        if "carbon" in name:color=rgb("#46505b");metal=.3
        pieces.append(Piece(v@R.T+m.geom_pos[i],n@R.T,np.asarray(f,np.int32),color,metal,rough,1,0,"robot",name,body))
    assert sum("qingyu_rviz_backbone" in a.name for a in pieces)==15
    assert sum("visual_gripper_finger" in a.name for a in pieces)==3
    assert any("visual_super_prop" in a.name for a in pieces)
    return pieces
