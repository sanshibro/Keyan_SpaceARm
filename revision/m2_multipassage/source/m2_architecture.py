"""Shared M2 industrial metalwork using the M0/M1 Piece/brick/turning library.

Every item is assigned to its existing collision enclosure. Recesses, folded
bezel shoulders, fasteners and tube grooves remove visual material inside the
registered solid; none projects into the certified free-space envelope.
"""
import numpy as np
from m0_scene import Piece,brick,turning,facets,basis,rgb
from m2_scene import boxes,pipes,GATES

TEAL='#447f7c';STEEL='#7f9297';DARK='#344e59';LIGHT='#c0cbca';AMBER='#b99164'

def architecture_parts():
    parts=[];ident=np.eye(3);bounds={name:(lo,hi) for name,lo,hi,_ in boxes()}
    def add(mesh,pos,frame,color,group,name,metal=.45,rough=.43,opacity=1,grid=0):
        v,n,f=mesh;parts.append(Piece(v@frame.T+pos,n@frame.T,f,color,metal,rough,opacity,grid,group,name))
    def box(name,lo,hi,color,group,bevel=.004,**kw):
        lo,hi=np.asarray(lo,float),np.asarray(hi,float)
        if np.any(hi<=lo):raise ValueError((name,lo,hi))
        add(brick(hi-lo,bevel),(lo+hi)/2,ident,color,group,name,**kw)
    def radial(name,profile,pos,axis,color,group,steps=64,**kw):
        add(turning(profile,steps),np.asarray(pos,float),basis(axis),color,group,name,**kw)
    # Low launch deck and raised gallery are two physically registered pieces.
    for group in ('entry_deck','deck','low_service_plinth'):
        lo,hi=bounds[group];z=hi[2];edge=.025
        box('structural plinth',lo+[.014,.014,.014],hi-[.014,.014,.065],'#8d9a9d',group,bevel=.012,metal=.22)
        box('machined tread plate',[lo[0],lo[1],z-.060],hi,'#aab8b5',group,bevel=.006,metal=.38,grid=1)
        box('front section rail',[lo[0]+.006,lo[1]+.0003,z-.046],[hi[0]-.006,lo[1]+.009,z-.032],DARK,group,bevel=.001)
        box('cut surface index',[lo[0]+.006,lo[1]+.001,z-.022],[hi[0]-.006,lo[1]+.006,z-.017],TEAL,group,bevel=.0006)
        if hi[0]-lo[0]>2:
            for x in np.arange(lo[0]+.4,hi[0]-.3,.72):
                box('recessed service panel',[x-.25,lo[1]+.001,lo[2]+.18],[x+.25,lo[1]+.007,z-.12],'#aebcba',group,bevel=.008)
                for j in range(6):
                    box('recessed ventilation cut',[x-.17+j*.055,lo[1]+.0001,z-.19],[x-.135+j*.055,lo[1]+.001,z-.18],DARK,group,bevel=.001)
    # Opaque rear panels, faint reader-side cut face, strong edge and welded ribs.
    for group in ('far_wall','near_guard','terminal_panel'):
        lo,hi=bounds[group];alpha=.045 if group=='near_guard' else (.50 if group=='far_wall' else .72)
        box('shared industrial skin',lo,hi,'#b6c5c6',group,bevel=.005,opacity=alpha,metal=.18,rough=.56)
        if group!='terminal_panel':
            for x in np.arange(lo[0]+.02,hi[0]-.01,.68):
                box('folded vertical frame',[x,lo[1]+.002,lo[2]+.01],[min(x+.014,hi[0]),hi[1]-.002,hi[2]-.01],
                    '#76908f' if group=='near_guard' else '#8ea6a5',group,bevel=.002,opacity=.30 if group=='near_guard' else .78)
            box('lower cut edge',[lo[0]+.004,lo[1]+.001,lo[2]],[hi[0]-.004,hi[1]-.001,lo[2]+.012],DARK,group,bevel=.001)
            box('upper cut edge',[lo[0]+.004,lo[1]+.001,hi[2]-.012],[hi[0]-.004,hi[1]-.001,hi[2]],DARK,group,bevel=.001,opacity=.16 if group=='near_guard' else 1.)
    for group in ('roof_edge_far','roof_edge_near'):
        lo,hi=bounds[group];alpha=.16 if group=='roof_edge_near' else 1.
        box('rolled roof channel',lo,hi,'#657c81',group,bevel=.006,metal=.60,opacity=alpha)
        box('channel recessed line',lo+[.004,.001,.014],hi-[.004,.001,.034],DARK,group,bevel=.001,opacity=alpha)

    for gate in (GATES[0],GATES[2]):
        x,y,z=gate['x'],gate['y'],gate['z'];half=gate['width']/2;low=z-gate['height']/2;high=z+gate['height']/2;x0=x-.055;x1=x+.055;prefix=gate['name']
        for suffix in ('sill','lintel','negative','positive'):
            group=prefix+'_'+suffix;lo,hi=bounds[group]
            box('folded bulkhead substrate',lo+[.012,0,0],hi-[.012,0,0],'#b0bfbe' if suffix=='sill' else '#b6c5c3',group,bevel=.003,metal=.40,rough=.47)
        def portal_layer(name,profile,color,metal=.55,rough=.4):
            def loop(d,ch,at):
                l,r=y-half-d,y+half+d;bot,top=low-d,high+d
                return np.array([[at,a,b] for a,b in ((l+ch,bot),(r-ch,bot),(r,bot+ch),(r,top-ch),(r-ch,top),(l+ch,top),(l,top-ch),(l,bot+ch))])
            rings=[loop(d,ch,xx) for d,ch,xx in profile];groups={}
            for a,b in zip(rings,rings[1:]+rings[:1]):
                for j in range(8):
                    k=(j+1)%8
                    for tri in (np.array([a[j],a[k],b[j]]),np.array([b[j],a[k],b[k]])):
                        if np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))<1e-11:continue
                        if tri[:,2].max()<=low+1e-9:key='sill'
                        elif tri[:,2].min()>=high-1e-9:key='lintel'
                        elif tri[:,1].max()<=y-half+1e-9:key='negative'
                        elif tri[:,1].min()>=y+half-1e-9:key='positive'
                        else:raise ValueError('Bezel intrudes into registered aperture: '+name)
                        groups.setdefault(prefix+'_'+key,[]).append(tri)
            alltri=np.asarray([t for ts in groups.values() for t in ts]);sign=np.einsum('ij,ij->',alltri[:,0],np.cross(alltri[:,1],alltri[:,2]))
            for group,tris in groups.items():
                v=np.vstack(tris);f=np.arange(len(v)).reshape(-1,3)
                if sign<0:f=f[:,::-1]
                add(facets(v,f),np.zeros(3),ident,color,group,name,metal=metal,rough=rough)
        # Tunnel liner is precisely on all four registered aperture planes.
        portal_layer('registered aperture liner',[(0,0,x0),(.008,.007,x0),(.008,.007,x1),(0,0,x1)],TEAL,.59)
        for face,sgn in ((x0,1),(x1,-1)):
            def xx(a):return face+sgn*a
            portal_layer('machined shoulder',[(.008,.006,xx(.010)),(.072,.032,xx(.010)),(.077,.036,xx(.016)),(.077,.036,xx(.027)),(.008,.006,xx(.027))],DARK,.64)
            portal_layer('folded satin bezel',[(.011,.009,xx(.008)),(.015,.012,xx(.002)),(.061,.029,xx(.002)),(.068,.032,xx(.009)),(.064,.030,xx(.011)),(.011,.009,xx(.011))],'#b6b5a9',.58)
            portal_layer('teal registered reveal',[(.003,.002,xx(.0002)),(.009,.007,xx(.0002)),(.009,.007,xx(.006)),(.003,.002,xx(.006))],TEAL,.50)
            fasteners=[(y+a,low-.038,'sill') for a in (-.30,-.14,.14,.30)]+[(y+a,high+.038,'lintel') for a in (-.30,-.14,.14,.30)]+[(y+ss*(half+.038),zz,'positive' if ss>0 else 'negative') for ss in (-1,1) for zz in (low+.10,z,high-.10)]
            for yy,zz,suffix in fasteners:
                group=prefix+'_'+suffix;pos=[xx(.0002),yy,zz]
                radial('flush countersunk ring',[(0,0),(.0063,0),(.0073,.001),(.0067,.003),(0,.003)],pos,[sgn,0,0],'#667c7c',group,steps=24,metal=.70)
                radial('recessed hex socket',[(0,0),(.0030,0),(.0030,.0004),(0,.0004)],[xx(.0001),yy,zz],[sgn,0,0],DARK,group,steps=6,metal=.25)
            for side in (-1,1):
                group=prefix+('_positive' if side>0 else '_negative');ya,yb=(y+half+.09,.785) if side>0 else (-.785,y-half-.09)
                if yb-ya>.07:
                    box('press formed web cassette',[min(xx(.012),xx(.037)),ya,low+.04],[max(xx(.012),xx(.037)),yb,high-.04],'#a1b2b2',group,bevel=.009)
                    for j in range(6):
                        zz=high-.23+.025*j
                        box('recessed web louvre',[min(xx(.0005),xx(.003)),ya+.018,zz],[max(xx(.0005),xx(.003)),yb-.018,zz+.006],DARK,group,bevel=.001)
                for j in range(3):
                    yy=y+side*(.292+j*.019)
                    box('sill index engraving',[min(xx(.0001),xx(.0011)),yy-.003,low-.055],[max(xx(.0001),xx(.0011)),yy+.003,low-.028],AMBER,prefix+'_sill',bevel=.0002)
            for j in range(29):
                zz=low+.065+j*.020
                box('metric reveal engraving',[min(xx(.0002),xx(.001)),y+half+.017,zz],[max(xx(.0002),xx(.001)),y+half+(.029 if j%5==0 else .023),zz+.001],DARK,prefix+'_positive',bevel=0)
        # Existing four structural bolts retain their own registered envelope.
        for side in (-1,1):
            for h in (1.48,2.68):
                group='%s_bolt_%s_%s'%(prefix,side,h);yy=y+side*(half+.08)
                radial('registered bulkhead fastener',[(0,0),(.009,0),(.012,.003),(.012,.125),(0,.128)],
                    [x-.064,yy,h],[1,0,0],'#6a7d81',group,steps=12,metal=.66)
    # Sheet-metal duct: co-planar inset panels and recessed louvres, no protruding lip.
    lo,hi=bounds['overhead_duct'];box('folded duct body',lo+[0,.010,0],hi-[0,.010,0],'#9baeb0','overhead_duct',bevel=.006,metal=.48)
    for x in (2.28,2.48,2.68,2.86):
        box('duct service cassette',[x-.067,lo[1]+.001,2.30],[x+.067,lo[1]+.006,2.61],'#b3bfbb','overhead_duct',bevel=.008)
        for j in range(6):
            zz=2.35+j*.031
            box('recessed duct louvre',[x-.048,lo[1]+.0002,zz],[x+.048,lo[1]+.0008,zz+.008],DARK,'overhead_duct',bevel=.001)
    for j,x in enumerate((2.19,2.37,2.56,2.75,2.93)):
        group='duct_rib_%d'%j;lo=np.array([x-.008,-.803,2.23]);hi=np.array([x+.008,.803,2.672])
        box('rolled duct seam',lo,hi,'#647e82',group,bevel=.002,metal=.62)
        box('seam cut face',[x-.005,-.803,2.232],[x+.005,-.800,2.670],TEAL,group,bevel=.001)
    for group,a,b,r,_ in pipes():
        if group in ('inspection_flange','flange_support'):continue
        axis=b-a;length=np.linalg.norm(axis);axis/=length
        # Turned outer face coincides with the physical cylinder except for grooves.
        grooves=[]
        for u in (.09,.42,.78,.94):
            h=min(.012,length*.08);s=min(length-h-.003,max(.003,u*length));grooves.append((u,s,h))
        profile=[(0,0),(r*.95,0),(r,.003)]
        for u,s,h in grooves:profile.extend([(r,s),(r-.0012,s),(r-.0012,s+h),(r,s+h)])
        profile.extend([(r,length-.003),(r*.95,length),(0,length)])
        radial('registered pipe skin',profile,a,axis,'#8c9d9d',group,steps=80,metal=.61,rough=.37)
        for u,s,h in grooves:
            radial('recessed coupling groove',[(r-.0012,s),(r-.00015,s),(r-.00015,s+h),(r-.0012,s+h)],a,axis,
                TEAL if u in (.42,.78) else DARK,group,steps=80,metal=.52)
            radial('machined groove index',[(r-.0004,s+h+.001),(r-.00005,s+h+.001),(r-.00005,s+h+.002),(r-.0004,s+h+.002)],a,axis,
                '#bfbbab',group,steps=80,metal=.68)
    return parts

def architecture_metadata():
    return dict(primitive_library='M0/M1 Piece, brick, turning, facets, basis',
      aperture_registration='Four inner faces preserved; all detail remains inside registered COL solids',
      pipe_registration='Native radii preserved; coupling grooves cut inward',
      launch_deck='Entry x<=0.60 m, top 0.55 m; raised main deck starts at x=0.60 m')
