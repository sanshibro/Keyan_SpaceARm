"""Offscreen GGX materials, filtered shadow map and metric section shading.
The only pose call in this pipeline is mj_kinematics on a saved qpos.
"""
from dataclasses import dataclass
import numpy as np
import moderngl as gl
import mujoco as mj
from PIL import Image
from m0_data import WORK,unit,matrix
from m0_scene import inspection_scene,robot_parts,rgb,C

@dataclass
class Camera:
    at:object
    toward_eye:object
    width:float
    def axes(self):
        back=unit(self.toward_eye)
        right=unit(np.cross([0,0,1],back))
        return right,np.cross(back,right),back
    def matrices(self,size):
        w,h=size;r,u,b=self.axes();focus=np.array(self.at)
        v=np.eye(4);v[:3,:3]=np.array([r,u,b]);v[:3,3]=-v[:3,:3]@(focus+5*b)
        height=self.width*h/w
        p=np.diag([2/self.width,2/height,-2/(12-.02),1.])
        p[2,3]=-(12+.02)/(12-.02)
        return p@v,focus+5*b
    def pixels(self,points,size):
        w,h=size;x=np.asarray(points)-self.at;r,u,_=self.axes()
        return np.stack([w/2+w*(x@r)/self.width,h/2-w*(x@u)/self.width],axis=-1)

HERO=Camera([1.00,.06,1.91],[.46,-1.,.49],3.38)
GATE_CAM=Camera([.98,.06,1.80],[.28,-1.,.15],.60)
DOCK_CAM=Camera([1.996,.249,1.716],[.32,-1.,.47],.58)
SIDE=Camera([.98,.035,1.99],[.13,-1.,.15],2.50)
LIGHT=Camera([1.03,.03,1.15],[.84,-1.02,1.65],4.5)

VERT="""#version 330
uniform mat4 vp, model, lightmat;
in vec3 pos,normal,rgb_in; in vec4 surface;
out vec3 wp,nn,col,lpos;out vec4 props;
void main(){vec4 world=model*vec4(pos,1.0);wp=world.xyz;nn=mat3(model)*normal;
col=rgb_in;props=surface;vec4 l=lightmat*world;lpos=l.xyz/l.w*.5+.5;gl_Position=vp*world;}
"""
SHADE="""#version 330
uniform sampler2D shadows;
uniform vec3 eye,lightdir;uniform int slice;
in vec3 wp,nn,col,lpos;in vec4 props;out vec4 frag;
const float pi=3.14159265;
float shadow(vec3 n){
 if(any(lessThan(lpos,vec3(0))) || any(greaterThan(lpos,vec3(1))))return 1.0;
 vec2 texel=1./vec2(textureSize(shadows,0));float sum=0.;
 float bias=.00014+.00047*(1.-abs(dot(n,lightdir)));
 for(int j=-2;j<=2;++j)for(int i=-2;i<=2;++i){
 vec2 o=vec2(i,j)*1.8*texel;sum+=step(lpos.z-bias,texture(shadows,lpos.xy+o).r);}
 return .44+.56*sum/25.;
}
float gridline(vec2 uv){
 vec2 f=abs(fract(uv+.5)-.5)/max(fwidth(uv),vec2(.00001));
 return 1.-smoothstep(.12,1.0,min(f.x,f.y));
}
vec3 direct(vec3 L,vec3 V,vec3 N,vec3 a,float m,float rough,vec3 energy){
 vec3 H=normalize(V+L);float nl=max(dot(N,L),0.),nv=max(dot(N,V),.002);
 float nh=max(dot(N,H),0.),vh=max(dot(V,H),0.);
 float aa=pow(rough,4.);float den=nh*nh*(aa-1.)+1.;
 float D=aa/(pi*den*den+.00001);
 float k=pow(rough+1.,2.)/8.;float G=(nv/(nv*(1.-k)+k))*(nl/(nl*(1.-k)+k));
 vec3 F0=mix(vec3(.044),a,m);vec3 F=F0+(1.-F0)*pow(1.-vh,5.);
 return energy*nl*((1.-F)*(1.-m)*a+pi*D*G*F/max(4.*nl*nv,.002));
}
void main(){if(slice==1 && wp.x>.949 && wp.x<1.051 && wp.y<-.22)discard;
vec3 N=normalize(nn),V=normalize(eye-wp);if(dot(N,V)<0.)N=-N;
 vec3 albedo=col;
 if(props.w>.5 && N.z>.88){
 float mi=gridline(wp.xy/.1), ma=gridline(wp.xy/.5);
 albedo=mix(albedo,vec3(.58,.65,.66),.27*mi);
 albedo=mix(albedo,vec3(.48,.55,.58),.43*ma);}
 vec3 a=pow(max(albedo,vec3(.001)),vec3(2.2));
 float metal=props.x,rough=clamp(props.y,.22,1.);
 vec3 R=reflect(-V,N);float sky=.52+.33*max(N.z,0.)+.09*max(N.y,0.);
 vec3 F0=mix(vec3(.044),a,metal);
 float strip=.20+.44*smoothstep(-.22,.65,R.z)+.46*pow(max(dot(R,normalize(vec3(-.4,-1.,.9))),0.),9.);
 vec3 env=mix(vec3(.87,.91,.96),vec3(1.04,1.01,.95),clamp(.45+.5*N.z,0.,1.));
 vec3 l=(a*(1.-metal)*sky+F0*strip)*env;
 l+=direct(lightdir,V,N,a,metal,rough,vec3(.89,.86,.78))*shadow(N);
 l+=direct(normalize(vec3(.8,.9,1.1)),V,N,a,metal,rough,vec3(.25,.29,.32));
 l=l/(vec3(1.)+.16*l);
 frag=vec4(clamp(pow(l*1.08,vec3(1./2.2)),0.,1.),props.z);}
"""
DEPTH_V="""#version 330
uniform mat4 vp,model;in vec3 pos;void main(){gl_Position=vp*model*vec4(pos,1.);}
"""
POST_V="""#version 330
out vec2 uv;void main(){vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));
uv=p;gl_Position=vec4(p*2.-1.,0.,1.);}
"""
POST_F="""#version 330
uniform sampler2D tex,depths;uniform float meters;in vec2 uv;out vec4 frag;
void main(){vec3 c=texture(tex,uv).rgb;float z=texture(depths,uv).r;
float occ=0.;vec2 px=1./vec2(textureSize(tex,0));
for(int j=0;j<12;++j){float a=6.2831853*(float(j)+.37)/12.;
 vec2 ofs=vec2(cos(a),sin(a))*px*(float(j%3)*2.+2.5)*clamp(2./meters,.65,2.5);
 float delta=(z-texture(depths,uv+ofs).r)*11.98;
 occ+=smoothstep(.002,.024,delta)*(1.-smoothstep(.04,.17,delta));}
frag=vec4(c*(1.-.115*occ/12.),1.);}
"""

def uniform_mat(p,name,mat):
    p[name].write(np.asarray(mat,dtype='f4').T.tobytes())

class DrawMesh:
    def __init__(self,ctx,program,shadow,parts):
        cols=[];indices=[];count=0
        for a in parts:
            n=len(a.v);row=np.empty((n,13),'f4')
            row[:,:3]=a.v;row[:,3:6]=a.n
            row[:,6:9]=rgb(a.color);row[:,9:]=[a.metal,a.rough,a.opacity,a.grid]
            cols.append(row);indices.append(a.f.astype('u4')+count);count+=n
        self.buf=ctx.buffer(np.ascontiguousarray(np.vstack(cols)).tobytes())
        self.indices=ctx.buffer(np.ascontiguousarray(np.vstack(indices)).tobytes())
        self.va=ctx.vertex_array(program,[(self.buf,'3f 3f 3f 4f','pos','normal','rgb_in','surface')],self.indices)
        self.shadow=ctx.vertex_array(shadow,[(self.buf,'3f 40x','pos')],self.indices)
        self.center=np.mean(np.vstack([a.v[::max(1,len(a.v)//96)] for a in parts]),axis=0)

class Renderer:
    def __init__(self,model):
        self.m=model;self.d=mj.MjData(model)
        self.ctx=c=gl.create_standalone_context(require=330)
        self.p=c.program(vertex_shader=VERT,fragment_shader=SHADE)
        self.s=c.program(vertex_shader=DEPTH_V,fragment_shader='#version 330\nvoid main(){}')
        self.post=c.program(vertex_shader=POST_V,fragment_shader=POST_F)
        self.screen=c.vertex_array(self.post,[])
        self.p['shadows'].value=0;self.post['tex'].value=0;self.post['depths'].value=1
        pieces,_=inspection_scene();native=robot_parts(model)
        self.wall=DrawMesh(c,self.p,self.s,[x for x in pieces if x.opacity>=1])
        self.glass=[DrawMesh(c,self.p,self.s,[x]) for x in pieces if x.opacity<1]
        self.robot={b:DrawMesh(c,self.p,self.s,[x for x in native if x.body==b])
                    for b in sorted({x.body for x in native})}
        self.names=[x.name for x in native]
        self.L,self.leye=LIGHT.matrices((2048,2048))
        self.depth=c.depth_texture((3072,3072));self.depth.compare_func=''
        self.depth.repeat_x=self.depth.repeat_y=False
        self.shadowbuf=c.framebuffer(depth_attachment=self.depth)
        self.targets={};self.transforms={}
        self.p['lightdir'].value=tuple(unit(LIGHT.toward_eye))
        uniform_mat(self.p,'lightmat',self.L)
        self.set_pose(None)
    def target(self,size):
        if size not in self.targets:
            col=self.ctx.texture(size,3);dep=self.ctx.depth_texture(size);dep.compare_func=''
            col.repeat_x=col.repeat_y=dep.repeat_x=dep.repeat_y=False
            out=self.ctx.texture(size,3)
            self.targets[size]=(col,dep,self.ctx.framebuffer([col],dep),self.ctx.framebuffer([out]))
        return self.targets[size]
    def set_pose(self,q):
        self.transforms={}
        if q is not None:
            self.d.qpos[:]=q;mj.mj_kinematics(self.m,self.d)
            self.transforms={b:matrix(self.d.xmat[b].reshape(3,3),self.d.xpos[b]) for b in self.robot}
        c=self.ctx;c.enable(gl.DEPTH_TEST);c.disable(gl.BLEND|gl.CULL_FACE)
        c.depth_func='<=';self.shadowbuf.use();self.shadowbuf.depth_mask=True
        self.shadowbuf.clear(depth=1);uniform_mat(self.s,'vp',self.L)
        uniform_mat(self.s,'model',np.eye(4));self.wall.shadow.render()
        for b,tr in self.transforms.items():
            uniform_mat(self.s,'model',tr);self.robot[b].shadow.render()
    def render(self,cam,size):
        c=self.ctx;col,dep,fb,out=self.target(size);fb.use();fb.depth_mask=True
        c.enable(gl.DEPTH_TEST);c.disable(gl.BLEND|gl.CULL_FACE)
        fb.clear(.979,.983,.98,1.,depth=1.)
        vp,eye=cam.matrices(size);uniform_mat(self.p,'vp',vp)
        self.p['eye'].value=tuple(eye);self.depth.use(0);uniform_mat(self.p,'model',np.eye(4))
        sectional=int(cam is GATE_CAM);self.p['slice'].value=sectional;self.wall.va.render()
        self.p['slice'].value=0
        for b,tr in self.transforms.items():
            uniform_mat(self.p,'model',tr);self.robot[b].va.render()
        uniform_mat(self.p,'model',np.eye(4))
        self.p['slice'].value=sectional
        c.enable(gl.BLEND|gl.CULL_FACE);c.blend_func=(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA)
        fb.depth_mask=False
        for part in sorted(self.glass,key=lambda x:float((x.center-eye)@unit(cam.toward_eye))):
            part.va.render()
        fb.depth_mask=True;c.disable(gl.BLEND|gl.DEPTH_TEST|gl.CULL_FACE)
        out.use();col.use(0);dep.use(1);self.post['meters'].value=cam.width
        self.screen.render(mode=gl.TRIANGLES,vertices=3)
        return Image.frombytes('RGB',size,out.read(components=3,alignment=1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

def ghost_picture(r,cam,run,times,size):
    r.set_pose(None);bg=np.array(r.render(cam,size)).astype(float);result=bg.copy()
    alpha=[.47,.63,.80,1.]
    for t,a in zip(times,alpha):
        r.set_pose(run.pose(t));pic=np.array(r.render(cam,size)).astype(float)
        weight=(np.max(abs(pic-bg),axis=2)>.55)[:,:,None]*a
        result=result*(1-weight)+pic*weight
    return Image.fromarray(np.uint8(np.clip(result,0,255)))
