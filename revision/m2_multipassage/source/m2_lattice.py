"""Causal layered whole-body search on observed geometry and the robot's shape space.

An arm configuration shares its rigid CAD transform across many translations
and yaws. Batching that work leaves time for multiple yaw and bending branches
before a longitudinal constriction. The beam is built across all local layers
and backtracked only after the furthest reachable layer is known.
"""
import numpy as np
from scipy.spatial import cKDTree
from m2_body_batch import BodyBatch
from m2_field import Field,RobotCloud

RAD = np.pi/180.

def angle(v):
    return (v + np.pi) % (2*np.pi) - np.pi

class LocalLattice:
    def __init__(self,kernel):
        self.k = kernel
        self.g = kernel.geometry
        self.cloud=RobotCloud(self.g)
        raw = np.array([(theta*RAD,phi*RAD) for theta in (35,55,72,85,98,112,117)
                        for phi in range(0,360,30)]+[(0.,np.pi)])
        self.static_shapes = self.feasible_shapes(raw)
        self.self_edges = {}
        # Robot-only self motion is independent of the map, flight and seed.
        self.transition_table(self.static_shapes)
        self.body_points=self.cloud.at('body',0.,0.)
        for s in self.static_shapes:self.cloud.at('arm',*s)

    def feasible_shapes(self,arr):
        kept=[]
        for shape in arr:
            d=self.g.self_report(*shape)
            if d['hazardous']>=self.k.settings.self_margin and d['assembly_min']>=0:
                kept.append(shape)
        return np.asarray(kept).reshape(-1,2)

    def self_edge(self,a,b):
        fa=a[1] if a[0]>.02 else b[1]
        fb=b[1] if b[0]>.02 else fa
        delta=angle(fb-fa)
        key=tuple(np.round([a[0],fa,b[0],fb],5))
        if key in self.self_edges:
            return self.self_edges[key]
        # Large moves are represented by intermediate lattice layers.
        ok=abs(a[0]-b[0])<.97 and (abs(delta)<1.65 or max(a[0],b[0])<.05)
        if ok:
            n=int(np.ceil(max(abs(delta)/.12,abs(a[0]-b[0])/.12)))
            for u in np.arange(1,n)/max(1,n):
                sr=self.g.self_report((1-u)*a[0]+u*b[0],fa+u*delta)
                if sr['hazardous']<self.k.settings.self_margin or sr['assembly_min']<0:
                    ok=False
                    break
        self.self_edges[key]=ok
        self.self_edges[(key[2],key[3],key[0],key[1])]=ok
        return ok

    def transition_table(self,shapes):
        result=np.ones((len(shapes),len(shapes)),dtype=bool)
        for i,a in enumerate(shapes):
            for j in range(i):
                result[j,i]=result[i,j]=self.self_edge(a,shapes[j])
        return result

    def body_layers(self,start,end,xs):
        precise=self.k.body_batch;points=self.body_points
        def query(states,cap=.162,reject_below=.085):
            return self.field.query(points,states,reject_below,cap)
        # Coordinates of the cloud are local to the current estimate. No world
        # gate, obstacle label, offline witness, or previous flight enters it.
        ys=np.unique(np.r_[np.arange(np.floor((start[1]-.76)/.075),np.ceil((start[1]+.76)/.075)+1)*.075,
                           start[1],end[1]])
        zs=np.unique(np.r_[np.arange(np.floor((start[2]-.50)/.055),np.ceil((start[2]+.50)/.055)+1)*.055,
                           start[2],end[2]])
        yz=np.array(np.meshgrid(ys,zs,indexing='ij')).reshape(2,-1).T
        base=np.zeros((len(xs),len(yz),6))
        base[:,:,0]=xs[:,None];base[:,:,1:3]=yz
        bg=query(base.reshape(-1,6)).reshape(base.shape[:2])
        tight=(bg.max(axis=1)<.117)|(np.sum(bg>=.106,axis=1)<4)
        near=np.min(np.where(tight[None,:],abs(xs[:,None]-xs[None,:]),np.inf),axis=1)
        expanded=(near<.66)
        yawbase=start[3]
        if abs(angle(yawbase))>.16:
            expanded|=xs<start[0]+.86
        if abs(angle(end[3]))>.16:
            expanded|=xs>end[0]-.48
        layers=[]
        for n,x in enumerate(xs):
            xyz=base[n]
            if expanded[n]:
                # Lateral and vertical shifts are searched together with body yaw.
                az=np.unique(np.round(angle(np.r_[np.arange(-180,180,30)*RAD,yawbase,end[3]]),5))
                az=az[abs(az)>1.e-4]
                extra=np.repeat(xyz,len(az),axis=0);extra[:,3]=np.tile(az,len(xyz))
                more=query(extra)
                pos=np.vstack((xyz,extra));gap=np.r_[bg[n],more]
                if near[n]<.27:
                    # Refine around every coarse yaw maximum. An asymmetric
                    # throat can admit only a narrow orientation interval.
                    best=[ids[np.argmax(gap[ids])] for aa in np.unique(pos[:,3])
                          for ids in [np.flatnonzero(pos[:,3]==aa)] if np.isfinite(gap[ids]).any()]
                    if not best:
                        raw=query(pos,cap=.12,reject_below=-.02);best=[ids[np.argmax(raw[ids])] for aa in np.unique(pos[:,3]) for ids in [np.flatnonzero(pos[:,3]==aa)]]
                    offsets=np.array(np.meshgrid([-.036,0.,.036],[-.025,0.,.025],[-12*RAD,0.,12*RAD])).reshape(3,-1).T
                    ref=np.repeat(pos[best],len(offsets),axis=0);ref[:,1:4]+=np.tile(offsets,(len(best),1))
                    pos=np.vstack((pos,ref));gap=np.r_[gap,query(ref)]
            else:
                pos=xyz;gap=bg[n]
            guide=start+(end-start)*((x-start[0])/max(.0001,end[0]-start[0]))
            selected=[]
            for aa in np.unique(pos[:,3]):
                ids=np.flatnonzero((pos[:,3]==aa)&(gap>=.08505))
                if not len(ids):continue
                loc=pos[ids,1:3]
                # Distinct clearance, lateral and vertical preferences retain
                # body choices on which a forward, side or resting arm can fit.
                ranked=2.4*np.maximum(0.,.135-gap[ids])+.045*np.linalg.norm(loc-guide[1:3],axis=1)
                dist=np.linalg.norm(loc-guide[1:3],axis=1)
                objectives=[ranked,ranked-.16*loc[:,1],ranked+.11*loc[:,1],
                            ranked-.09*loc[:,0],ranked+.09*loc[:,0],dist+.8*np.maximum(0.,.112-gap[ids])]
                chosen=np.unique([ids[np.argmin(j)] for j in objectives])
                selected.extend(chosen)
            if not selected:
                layers.append((np.empty((0,6)),np.empty(0)));continue
            ids=np.asarray(selected,int)
            chosen=pos[ids]
            if self.k.warm_points is not None and self.k.warm_points[0,0]<=x<=self.k.warm_points[-1,0]:
                warm=np.array([np.interp(x,self.k.warm_points[:,0],self.k.warm_points[:,j]) for j in range(6)])
                chosen=np.vstack((chosen,warm));self.k.warm_used+=1
            exact=precise.query(chosen,cap=.16,reject_below=.085);ok=exact>=.08505
            layers.append((chosen[ok],exact[ok]))
        return layers

    def search(self,start,end,final=False):
        c=self.k.settings;method=self.k.method
        n=max(7,int(np.ceil((end[0]-start[0])/.095))+1)
        xs=np.linspace(start[0],end[0],n)
        self.field=Field(self.g.snapshot,start,end)
        bodies=self.body_layers(start,end,xs[1:])
        if final:bodies[-1]=(end.reshape(1,6),self.k.body_batch.query(end.reshape(1,6)))
        if method=='sequential':
            # Fix an independently chosen body route before selecting arm states.
            bodies=self.sequential_bodies(start,bodies)
        qbody=np.vstack([p for p,g in bodies])
        bg=np.concatenate([g for p,g in bodies]);layer_id=np.repeat(np.arange(1,n),[len(p) for p,g in bodies])
        if method=='straight':shapes=np.array([[0.,start[5]]])
        elif method=='fixed':shapes=np.array([[80*RAD,np.pi]])
        else:
            shapes=np.unique(np.vstack((self.static_shapes,self.feasible_shapes(np.vstack((start[4:],end[4:]))))),axis=0)
        start_shape=int(np.argmin(np.linalg.norm(shapes[:,0,None]*np.column_stack((np.cos(shapes[:,1]),np.sin(shapes[:,1])))-
            start[4]*np.array([np.cos(start[5]),np.sin(start[5])]),axis=1)))
        transitions=self.transition_table(shapes)
        shape_com=np.array([np.linalg.norm(self.g.local(float(s[0]),float(s[1]))[2][:2]) for s in shapes])
        armg=np.empty((len(shapes),len(bg)))
        for j,shape in enumerate(shapes):
            armg[j]=self.field.query(self.cloud.at('arm',*shape),qbody,cap=.16,reject=.085)
        data=[(start.reshape(1,6),np.array([start_shape]),np.array([.14]),np.array([.14]))]
        for i in range(1,n):
            ids=np.flatnonzero(layer_id==i)
            sj,bj=np.nonzero(armg[:,ids]>=.08505);bi=ids[bj]
            q=qbody[bi].copy();q[:,4:]=shapes[sj]
            if final and i==n-1:
                sel=np.isclose(q[:,4],end[4],atol=1.e-6)&(abs(angle(q[:,5]-end[5]))<1.e-6)
                q,sj,bi=q[sel],sj[sel],bi[sel]
            data.append((q,sj,bg[bi],armg[sj,bi]))
        return self.connected(start,end,data,transitions,shape_com,final)

    @staticmethod
    def sequential_bodies(start,bodies):
        prev=start[:4];out=[]
        for p,g in bodies:
            if not len(p):out.append((p,g));continue
            cost=np.sum((p[:,1:3]-prev[1:3])**2,axis=1)+.08*angle(p[:,3]-prev[3])**2+4.*np.maximum(0.,.123-g)**2
            ix=np.argmin(cost);prev=p[ix,:4];out.append((p[ix:ix+1],g[ix:ix+1]))
        return out

    @staticmethod
    def features(q):
        return np.column_stack((3.6*q[:,1],4.4*q[:,2],.58*np.cos(q[:,3]),.58*np.sin(q[:,3]),
                               .34*q[:,4]*np.cos(q[:,5]),.34*q[:,4]*np.sin(q[:,5])))

    def connected(self,start,end,data,transitions,shape_com,final):
        k=self.k;cost=np.zeros(1);prevq=data[0][0];prevs=data[0][1];kept=[prevq]
        pred=[];diag=[];chosen_cost=[];prev_feature=self.features(prevq)
        for layer,(q,s,body,arm) in enumerate(data[1:],1):
            if not len(q):break
            features=self.features(q)
            nn=min(32,len(cost))
            ds,j=cKDTree(prev_feature).query(features,k=nn)
            if nn==1:ds,j=ds[:,None],j[:,None]
            step_cost=ds**2
            attainable=transitions[prevs[j],s[:,None]]&(abs(angle(q[:,None,3]-prevq[j,3]))<=75*RAD)
            step_cost[~attainable]=np.inf
            gap=np.minimum(body,arm)
            node=90.*np.maximum(.126-gap,0.)**2 + 2.*np.maximum(.147-gap,0.)
            node+=.033*np.linalg.norm(q[:,1:3]-start[1:3],axis=1)**2 + .0025*q[:,4]**2
            if k.coupled:node+=1.4*shape_com[s]**2
            scores=cost[j]+step_cost
            best=np.argmin(scores,axis=1);row=np.arange(len(q))
            total=scores[row,best]+node;previous=j[row,best]
            sel=np.flatnonzero(np.isfinite(total))
            if not len(sel):break
            # Multi-plane beam: stratify by body yaw and Cartesian bend sector
            # so a newly visible constriction can draw on a side-bending branch.
            limit=250
            if len(sel)>limit:
                ranked=sel[np.argsort(total[sel])]
                bins=np.column_stack((np.rint(q[ranked,3]/(15*RAD)),
                    np.rint(q[ranked,1]/.09),np.rint(q[ranked,4]*np.cos(q[ranked,5])/.48),
                    np.rint(q[ranked,4]*np.sin(q[ranked,5])/.48))).astype(int)
                unique=np.sort(np.unique(bins,axis=0,return_index=True)[1])
                sel=ranked[np.r_[unique,np.setdiff1d(np.arange(len(ranked)),unique,assume_unique=True)][:limit]]
            pred.append(previous[sel]);kept.append(q[sel]);prevq=q[sel];prevs=s[sel]
            prev_feature=features[sel];cost=total[sel];chosen_cost.append(cost)
            diag.append(dict(x=float(q[0,0]),eligible=len(q),survivors=len(sel)))
        k.seed_diagnostics=diag
        if len(kept)<4:raise ValueError('No connected local whole-body seed, '+str(len(kept)-1)+' supported layers')
        i=int(np.argmin(cost));path=[]
        for j in range(len(kept)-1,0,-1):
            path.append(kept[j][i]);i=pred[j-1][i]
        route=np.array([start]+path[::-1])
        # Select a consistent chart while crossing the zero-curvature pole.
        rotating=np.flatnonzero(route[:,4]>.02)
        if len(rotating):
            route[:rotating[0],5]=route[rotating[0],5]
            for j in range(rotating[-1]+1,len(route)):route[j,5]=route[rotating[-1],5]
        route[:,3]=np.unwrap(route[:,3]);route[:,5]=np.unwrap(route[:,5])
        return route,diag,len(kept)==len(data)
