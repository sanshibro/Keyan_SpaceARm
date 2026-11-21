"""Live RGB-D -> native ORB-SLAM3 -> frozen 881a799 AnchoredRGBD.
Only SensorCanvas consumes simulated qpos. OpticalEstimator consumes pixels,
native optical poses and the caller's PX4 attitude; no plant/command/truth access.
"""
import argparse,csv,hashlib,json,os,re,select,shlex,subprocess,time
from pathlib import Path
from xml.etree import ElementTree as ET
import cv2,numpy as np
from scipy.spatial.transform import Rotation
from anchored_rgbd import AnchoredRGBD
from round_aug_vis import body_rotation,bend
ROOT=Path(__file__).resolve().parents[2]
FREEZE='881a799'
DATA=ROOT/'revision/data/round_gate_feedback'
FROZEN=['mujoco_closed_loop/anchored_rgbd.py','revision/scripts/round_aug_vis.py','revision/scripts/round_aug_vis_cov.py']
cv2.setNumThreads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def writej(p,data):Path(p).write_text(json.dumps(clean(data),indent=2,allow_nan=False)+'\n')
def frozen_manifest():
    result={}
    for relative in FROZEN:
        expected=subprocess.check_output(['git','show',FREEZE+':'+relative],cwd=ROOT)
        digest=hashlib.sha256(expected).hexdigest();actual=sha(ROOT/relative)
        if actual!=digest:raise RuntimeError('Frozen perception source modified: '+relative)
        result[relative]=actual
    return result
def build_native():
    orb=Path.home()/'桌面/Test/ORB_SLAM3';folder=DATA/'native';folder.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().with_name('gate_live_orb.cc');binary=folder/'gate_live_orb'
    sentinel=folder/'build.json'
    if not binary.exists() or not sentinel.exists() or json.loads(sentinel.read_text())['source_sha256']!=sha(source):
        build=orb/'build';flagsfile=build/'CMakeFiles/rgbd_tum.dir/flags.make';lines=flagsfile.read_text().splitlines();flags=[]
        for key in ('CXX_FLAGS','CXX_DEFINES','CXX_INCLUDES'):
            flags+=shlex.split(next(s.partition('=')[2] for s in lines if s.startswith(key+' ')))
        link=shlex.split((build/'CMakeFiles/rgbd_tum.dir/link.txt').read_text())
        libs=link[next(i for i,s in enumerate(link) if s.startswith('-Wl,-rpath')):]
        subprocess.run([link[0],*flags,'-w',str(source),'-o',str(binary),*libs],cwd=build,check=True)
        writej(sentinel,dict(source_sha256=sha(source),binary_sha256=sha(binary),
          library_sha256=sha(orb/'lib/libORB_SLAM3.so'),freeze=FREEZE))
    return binary,orb

class NativeTracker:
    def __init__(self,settings,folder):
        binary,orb=build_native();self.folder=Path(folder);self.log=(self.folder/'native.log').open('w')
        self.process=subprocess.Popen([str(binary),str(orb/'Vocabulary/ORBvoc.txt'),str(settings),str(self.folder/'native.csv')],
           cwd=self.folder,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        try:self._read('READY',60.)
        except BaseException:
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            self.log.close();raise
        self.fields=['t','state','map_id','matched','features','reproj_px','track_ms','occluded']+['r'+str(i) for i in range(12)]+['atlas_id','origin_kf']
    def _read(self,prefix,timeout):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            readable,_,_=select.select([self.process.stdout],[],[],min(.5,max(0.,deadline-time.monotonic())))
            if not readable:
                if self.process.poll() is not None:raise RuntimeError('native ORB exited; see native.log')
                continue
            line=self.process.stdout.readline().strip()
            if line.startswith(prefix):return line
            if line.startswith('ERROR'):raise RuntimeError(line)
            if line:self.log.write('stdout: '+line+'\n');self.log.flush()
        raise TimeoutError('Native ORB response: '+prefix)
    def update(self,frame,t,rgb,depth):
        self.process.stdin.write('FRAME %d %.9f %s %s\n'%(frame,t,Path(rgb).resolve(),Path(depth).resolve()));self.process.stdin.flush()
        line=self._read('POSE ',30.);_,idx,row=line.split(' ',2)
        if int(idx)!=frame:raise RuntimeError('Native response frame mismatch')
        values=[float(v) for v in row.split(',')]
        return dict(zip(self.fields,values))
    def close(self):
        if self.process.poll() is None:
            try:self.process.stdin.write('QUIT\n');self.process.stdin.flush();self.process.wait(timeout=30)
            except (BrokenPipeError,subprocess.TimeoutExpired):
                self.process.terminate()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.log.close()

class OpticalEstimator:
    """Frozen geometry and covariance, with only a constant input-clock translation."""
    def __init__(self,K,calibration):
        self.engine=AnchoredRGBD(K,1000.)
        self.UB=np.array(calibration['T_UB'])[:3,:3];self.TC=np.array(calibration['T_TC'])[:3,:3]
        self.WM=None;self.first_t=None;self.stable=0;self.lock_at=-1e6;self.last_map=None;self.frame=0
    def update(self,t,gray,depth,native,rpy_frd_ned):
        i=self.frame;self.frame+=1;cv2.setRNGSeed(31071+i)
        if self.first_t is None:self.first_t=float(t)
        clock=12.+float(t)-self.first_t
        feat=self.engine.features(gray,depth)
        pose=np.eye(4);pose[:3]=np.array([native['r'+str(j)] for j in range(12)]).reshape(3,4)
        ok=bool(native['state']==2 and native['matched']>=20 and np.isfinite(native['reproj_px']) and native['reproj_px']<3. and native['map_id']>=0)
        corrected,_,diag=self.engine.update(clock,pose,int(native['map_id']),ok,feat,i)
        base=body_rotation(rpy_frd_ned)
        if self.WM is None and self.engine.initial is not None:
            self.WM=base@self.UB@self.TC@self.engine.initial[1][:3,:3]
        b=np.full(2,np.nan);R=np.full((2,2),np.nan);q=0.;valid=corrected is not None;ready=False
        if valid:
            BT=(base@self.UB).T@self.WM@corrected[:3,:3].T@self.TC.T;b=bend(BT)
            J=np.empty((2,3))
            for a in range(3):
                v=np.zeros(3);v[a]=1e-5;J[:,a]=(bend(Rotation.from_rotvec(v).as_matrix()@BT)-b)/1e-5
            if 'cov_map' in diag:
                A=(base@self.UB).T@self.WM;cov=A@np.array(diag['cov_map'])@A.T
                R=J@cov@J.T+np.eye(2)*np.deg2rad(.2)**2
            else:R=np.deg2rad(diag['sigma_deg'])**2*(J@J.T)+np.eye(2)*np.deg2rad(.15)**2
            if diag.get('relocked',False) or (self.last_map is not None and diag['map']!=self.last_map):self.lock_at=clock
            self.last_map=diag['map']
            depth_ratio=float(diag.get('depth_ratio',.2));rate=float(diag.get('angular_rate_deg_s',0.))
            extent=min(1.,min(np.array(diag.get('extent',[75.,48.]))/np.array([260.,180.])))
            factor=1./max(.08,depth_ratio)*(1+(rate/90.)**2)
            if diag.get('plane_rank',0)<3:factor*=1.+max(0.,.3-extent)/.3
            factor*=1.+3.*max(0.,1.-(clock-self.lock_at)/.32);R*=factor
            q=float(np.clip(np.deg2rad(.85)**2/float(np.linalg.eigvalsh(R).max()),0,1))
            self.stable=self.stable+1 if q>.20 and diag.get('age_s',100)<.3 else 0;ready=self.stable>=4
        else:self.stable=0
        diag['algorithm_t']=clock;diag['t']=float(t)
        return dict(t=float(t),arrival_t=float(t)+.2,frame=i,b=b,R=R,valid=bool(valid),locked=bool(valid),
                    quality=q,ready=bool(ready),diagnostics=diag)

def textured_gate_box(asset,world,source,folder,seed):
    """Map the gate finish onto its actual six faces, including vertical faces."""
    from round_wrist_render import surface_image
    name=source.get('name')
    mesh='gate_skin_'+name
    half=np.fromstring(source.get('size'),sep=' ')
    # Explicit UVs prevent a vertical gate face sampling a single texture line.
    vertices=[(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
              (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
    faces=[(1,5,8,4),(2,3,7,6),(1,2,6,5),(4,8,7,3),(1,4,3,2),(5,6,7,8)]
    shape_key=hashlib.sha256(half.tobytes()).hexdigest()[:10]
    obj=folder/(mesh+'-uv1-'+shape_key+'.obj')
    lines=['v %d %d %d'%v for v in vertices]
    for i,face in enumerate(faces):
        vertices_world=np.asarray([vertices[k-1] for k in face])*half
        lengths=np.array([np.linalg.norm(vertices_world[1]-vertices_world[0]),
                          np.linalg.norm(vertices_world[2]-vertices_world[1])])
        scale=.80*lengths/lengths.max()
        offset=np.array([.10*(i%3)/2.,.10*(i//3)])
        for uv in ([0,0],[1,0],[1,1],[0,1]):
            tex=offset+scale*np.array(uv)
            lines.append('vt %.8f %.8f'%tuple(tex))
        idx=['%d/%d'%(v,i*4+j+1) for j,v in enumerate(face)]
        lines.extend(('f '+' '.join(idx[:3]),'f '+' '.join([idx[0],idx[2],idx[3]])))
    if not obj.exists():obj.write_text('\n'.join(lines)+'\n')
    tex=folder/(mesh+'-finish.png')
    if not tex.exists():surface_image(tex,seed)
    ET.SubElement(asset,'texture',name=mesh+'_tex',type='2d',file=str(tex.resolve()))
    ET.SubElement(asset,'material',name=mesh+'_mat',texture=mesh+'_tex',specular='.03',shininess='.04')
    ET.SubElement(asset,'mesh',name=mesh,file=str(obj.resolve()),scale=source.get('size'))
    ET.SubElement(world,'geom',name='live_sensor_'+name,type='mesh',mesh=mesh,pos=source.get('pos'),
                  contype='0',conaffinity='0',group='5',material=mesh+'_mat',rgba='.75 .77 .77 1')

def add_sensor_canvas(root,folder=None,texture_seed=91105):
    """Add ONLY group-5 visual surfaces. Physics world geoms/materials stay untouched.
    Invoke at the END of scene_update, before compiling the one shared MjModel.
    Observers set geomgroup[5]=0; the wrist sensor sets geomgroup[5]=1.
    """
    from round_wrist_render import surface_image
    folder=Path(folder) if folder is not None else DATA/'canvas_assets'
    folder.mkdir(parents=True,exist_ok=True);asset=root.find('asset');world=root.find('worldbody')
    if world.find("geom[@name='live_finish_far']") is not None:raise ValueError('Sensor canvas already installed')
    for j,kind in enumerate(('floor','far','left','right','rear','gate')):
        tex=folder/('finish-'+kind+'.png')
        if not tex.exists():surface_image(tex,texture_seed+j)
        ET.SubElement(asset,'texture',name='live_tex_'+kind,type='2d',file=str(tex.resolve()))
        ET.SubElement(asset,'material',name='live_mat_'+kind,texture='live_tex_'+kind,
                      texrepeat='1 1',specular='.04',shininess='.05',texuniform='false')
    original=list(world.findall('geom'))
    floor=world.find("geom[@name='floor']")
    if floor is not None:
        copy=ET.fromstring(ET.tostring(floor));copy.set('name','live_sensor_floor');copy.set('group','5')
        copy.set('contype','0');copy.set('conaffinity','0');copy.set('material','live_mat_floor');copy.set('rgba','.94 .95 .96 1')
        position=np.array([float(v) for v in copy.get('pos','0 0 0').split()]);position[2]+=.0001
        copy.set('pos',' '.join(map(str,position)));world.append(copy)
    for index,g in enumerate(original):
        name=g.get('name','')
        if g.get('type')=='box' and (int(g.get('contype','1'))!=0 or name.startswith(('G1_','gate_','GF_','gatefeedback_','feedback_'))):
            textured_gate_box(asset,world,g,folder,texture_seed+41*index)
    for kind,pos,axis,size in [
        ('far','3.35 0 1.8','0 1 0 -1.5707963','1.9 2.7 .012'),
        ('rear','-2.7 0 1.8','0 1 0 1.5707963','1.9 2.7 .012'),
        ('left','.3 -2.6 1.8','1 0 0 -1.5707963','3.1 1.9 .012'),
        ('right','.3 2.6 1.8','1 0 0 1.5707963','3.1 1.9 .012')]:
        ET.SubElement(world,'geom',name='live_finish_'+kind,type='box',pos=pos,axisangle=axis,size=size,
                      contype='0',conaffinity='0',material='live_mat_'+kind,group='5')

class SensorCanvas:
    """Simulation-only sensor boundary. Shared immutable MjModel, separate MjData."""
    def __init__(self,plant,folder,texture_seed=91105):
        import mujoco
        from sensors import RGBDSensor
        self.mujoco=mujoco
        try:plant.m.geom('live_finish_far')
        except KeyError:raise ValueError('Call add_sensor_canvas(root) at the end of scene_update before constructing the plant')
        self.model=plant.m;self.data=mujoco.MjData(self.model)
        self.sensor=RGBDSensor(self.model,'wrist_rgbd',480,360);self.K=self.sensor.K.copy()
        self.scene_option=mujoco.MjvOption();self.scene_option.geomgroup[5]=1
        # Group 4 contains physics/audit proxies; copy surfaces are in sensor-only group 5.
        self.scene_option.geomgroup[4]=0
        self.nq=self.model.nq
        (Path(folder)/'sensor-canvas.xml').write_text(plant.xml)
    def capture(self,t,qpos):
        self.data.qpos[:]=np.asarray(qpos,dtype=float).copy();self.data.time=float(t)
        self.mujoco.mj_forward(self.model,self.data);r=self.sensor.renderer
        r.update_scene(self.data,camera='wrist_rgbd',scene_option=self.scene_option)
        rgb=r.render().copy();r.enable_depth_rendering();depth=r.render().copy();r.disable_depth_rendering()
        depth[(depth<.15)|(depth>6)]=np.nan
        return cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),np.rint(1000*np.nan_to_num(depth,nan=0.)).astype('uint16')
    def close(self):self.sensor.close()

class LiveOptics:
    """capture(t,current_qpos,current_px4_rpy,blackout_start) returns exposure-stamped observation.
    Calling before the next 25 Hz exposure returns None. The caller owns delivery at t+0.20.
    Start in the calibrated neutral shape. Default output convention is case_folder/'optics'.
    """
    def __init__(self,plant,folder,condition='natural',texture_seed=91105):
        self.folder=Path(folder).resolve();self.folder.mkdir(parents=True,exist_ok=True)
        if (self.folder/'frames.txt').exists():raise FileExistsError('Use a new run folder; optical streams are not overwritten')
        self.condition={'clear':'natural','blackout':'occluded'}.get(condition,condition)
        if self.condition not in ('natural','occluded','graded'):raise ValueError('Unknown optical condition')
        self.frozen=frozen_manifest();self.canvas=getattr(self, 'canvas_class', SensorCanvas)(plant,self.folder,texture_seed)
        # Fixed archived calibration only: no previous-run image, pose, command, or observation.
        calibrated=ROOT/'revision/data/round_shape_clearance/pilot/241118-online_pause/packets.npz'
        with np.load(str(calibrated)) as c:calibration={k:c[k].tolist() for k in ('T_UB','T_TC')}
        writej(self.folder/'fixed-calibration.json',calibration)
        self.estimator=OpticalEstimator(self.canvas.K,calibration)
        orb=Path.home()/'桌面/Test/ORB_SLAM3';tmpl=(orb/'Examples/RGB-D/TUM1.yaml').read_text()
        K=self.canvas.K
        values={'Camera1.fx':K[0,0],'Camera1.fy':K[1,1],'Camera1.cx':K[0,2],'Camera1.cy':K[1,2],
                'Camera.width':480,'Camera.height':360,'Camera.fps':25,'Camera.RGB':0,'RGBD.DepthMapFactor':1000.,
                **{'Camera1.'+k:0. for k in ('k1','k2','k3','p1','p2')}}
        for k,v in values.items():
            tmpl,n=re.subn('^'+re.escape(k)+r':.*$',k+': '+str(v),tmpl,flags=re.M)
            if n!=1:raise ValueError('ORB settings field '+k)
        settings=self.folder/'settings.yaml';settings.write_text(tmpl)
        try:self.native=NativeTracker(settings,self.folder)
        except BaseException:self.canvas.close();raise
        self.pixels=self.folder/'pixels';self.pixels.mkdir(exist_ok=True)
        self.index=(self.folder/'index.jsonl').open('w');self.frames=(self.folder/'frames.txt').open('w')
        self.log=(self.folder/'optical.jsonl').open('w')
        self.next_exposure_t=None;self.last_t=None;self.rows=[];self.closed=False;self.perf=[]
        writej(self.folder/'manifest.json',dict(freeze=FREEZE,frozen_sources=self.frozen,condition=self.condition,
          frequency_hz=25,delivery_delay_s=.2,canvas_width=480,canvas_height=360,texture_seed=texture_seed,
          scene_physics_modified=False,calibration_source='fixed T_UB/T_TC only, neutral initialization',
          input_boundary='sensor canvas: current qpos; estimator: current pixels, native pose, supplied PX4 attitude',
          native_binary_sha256=sha(DATA/'native/gate_live_orb'),source_sha256=sha(__file__),
          clock_translation='AnchoredRGBD t=12+(exposure-first exposure), preserves frozen 2-second initial bank window'))
    def due(self,t):return self.next_exposure_t is None or t>=self.next_exposure_t-1e-7
    def capture(self,t,sim_qpos,rpy_frd_ned,blackout_start=None):
        if self.closed:raise RuntimeError('Optical stream closed')
        t=float(t)
        if not self.due(t):return None
        if self.last_t is not None and t<=self.last_t:raise ValueError('Exposure time must strictly increase')
        start=time.perf_counter();frame=len(self.rows)
        rgb,depth=self.canvas.capture(t,sim_qpos);rendered=time.perf_counter()
        fraction=0.
        if self.condition!='natural' and blackout_start is not None:
            d=t-float(blackout_start)
            if -1.6<=d<0:fraction=.68*float(np.clip((d+1.6)/1.6,0,1))
            if self.condition=='occluded' and 0<=d<4:fraction=1.
        width=round(rgb.shape[1]*fraction)
        if width:rgb[:,:width]=0;depth[:,:width]=0
        rp=self.pixels/('r%06d.jpg'%frame);dp=self.pixels/('d%06d.png'%frame)
        assert cv2.imwrite(str(rp),rgb,[cv2.IMWRITE_JPEG_QUALITY,92])
        assert cv2.imwrite(str(dp),depth,[cv2.IMWRITE_PNG_COMPRESSION,1])
        # Both native and frozen anchor geometry consume the identical decoded saved pixels.
        gray=cv2.imread(str(rp),0);measured_depth=cv2.imread(str(dp),cv2.IMREAD_UNCHANGED)
        native=self.native.update(frame,t,rp,dp);tracked=time.perf_counter()
        record=self.estimator.update(t,gray,measured_depth,native,np.asarray(rpy_frd_ned,dtype=float))
        done=time.perf_counter();record['processing_wall_s']=done-start
        self.frames.write('%.9f %s %s\n'%(t,rp.relative_to(self.folder),dp.relative_to(self.folder)));self.frames.flush()
        self.index.write(json.dumps(dict(frame=frame,t=t,arrival_t=t+.2,rgb=str(rp.relative_to(self.folder)),
             depth=str(dp.relative_to(self.folder)),rgb_sha256=sha(rp),depth_sha256=sha(dp),mask_fraction=fraction,
             rpy_frd_ned=list(map(float,rpy_frd_ned)),sensor_qpos_sha256=hashlib.sha256(np.asarray(sim_qpos,dtype=np.float64).tobytes()).hexdigest()))+'\n');self.index.flush()
        record['native']=native;self.log.write(json.dumps(clean(record),allow_nan=False)+'\n');self.log.flush()
        self.rows.append(record);self.perf.append([rendered-start,tracked-rendered,done-tracked,done-start])
        self.last_t=t;self.next_exposure_t=t+.04
        return record
    expose=capture
    update=capture
    def close(self):
        if self.closed:return
        self.closed=True
        try:self.native.close()
        finally:
            self.canvas.close();self.index.close();self.frames.close();self.log.close()
        np.savez_compressed(self.folder/'observations.npz',
          **{k:np.array([r[k] for r in self.rows]) for k in ('t','arrival_t','b','R','valid','locked','quality','ready')})
        perf=np.array(self.perf)
        writej(self.folder/'complete.json',dict(frames=len(self.rows),valid=sum(r['valid'] for r in self.rows),
          processing_wall_median_ms=(np.median(perf,axis=0)*1000).tolist() if len(perf) else [],
          processing_wall_p95_ms=(np.percentile(perf,95,axis=0)*1000).tolist() if len(perf) else [],
          timing_columns=['render','native_including_disk','anchored_geometry','total'],
          native_csv_sha256=sha(self.folder/'native.csv'),index_sha256=sha(self.folder/'index.jsonl'),
          observations_sha256=sha(self.folder/'observations.npz'),frozen_sources_unchanged=frozen_manifest()==self.frozen))
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def smoke(folder,count=200):
    import mujoco
    from round_f_plant import Plant
    from arm_command import ArmCommand
    plant=Plant(scene_update=add_sensor_canvas);initial=plant.d.qpos.copy();input_hash=hashlib.sha256(plant.xml.encode()).hexdigest()
    with LiveOptics(plant,folder,'occluded') as sensor:
        for i in range(count):
            t=i*.04
            # A fresh fixed-base physical test; no past trajectory or images are loaded.
            theta=np.deg2rad(25)*np.clip((t-2)/2,0,1);phi=np.pi-np.deg2rad(25)*np.clip((t-3)/2,0,1)
            plant.arm.apply(ArmCommand(i,t,float(theta),float(phi)))
            for _ in range(80):mujoco.mj_step(plant.m,plant.d)
            mujoco.mj_forward(plant.m,plant.d)
            sensor.capture(t,plant.d.qpos.copy(),[0.,0.,np.pi/2],blackout_start=2.)
            if i%25==0:print('SMOKE',i,t,flush=True)
        assert hashlib.sha256(plant.xml.encode()).hexdigest()==input_hash
    z=np.load(Path(folder)/'observations.npz');dark=(z['t']>=2)&(z['t']<6)
    assert not z['valid'][dark].any();assert np.allclose(z['arrival_t']-z['t'],.2)
    writej(Path(folder)/'smoke-audit.json',dict(new_physics_run=True,no_px4_started=True,frames=len(z['t']),
      blackout_valid=int(z['valid'][dark].sum()),delivery_delay_exact=True,physics_xml_unchanged=True,
      frame_interval_max_error_s=float(np.max(abs(np.diff(z['t'])-.04)))))
    print((Path(folder)/'complete.json').read_text(),flush=True)

def verify_stream(folder,n=175):
    """Reprocess a prefix of this newly acquired stream; native poses stay sealed."""
    folder=Path(folder);index=[json.loads(s) for s in (folder/'index.jsonl').read_text().splitlines()]
    with (folder/'native.csv').open() as f:native=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
    full=np.load(folder/'observations.npz');assert len(index)==len(native)==len(full['t'])
    n=min(n,len(index));fs=cv2.FileStorage(str(folder/'settings.yaml'),cv2.FILE_STORAGE_READ)
    f=lambda k:fs.getNode(k).real();K=np.array([[f('Camera1.fx'),0,f('Camera1.cx')],[0,f('Camera1.fy'),f('Camera1.cy')],[0,0,1.]])
    fs.release();calibration=json.loads((folder/'fixed-calibration.json').read_text())
    estimator=OpticalEstimator(K,calibration);rows=[];black=0
    for i in range(n):
        entry=index[i];rgb=folder/entry['rgb'];depth=folder/entry['depth']
        assert sha(rgb)==entry['rgb_sha256'] and sha(depth)==entry['depth_sha256']
        gray=cv2.imread(str(rgb),0);dep=cv2.imread(str(depth),cv2.IMREAD_UNCHANGED)
        if entry['mask_fraction']==1:
            assert not np.any(gray) and not np.any(dep);black+=1
        row=estimator.update(entry['t'],gray,dep,native[i],entry['rpy_frd_ned']);rows.append(row)
    fields={}
    for k in ('t','arrival_t','b','R','valid','locked','quality','ready'):
        a=full[k][:n];b=np.array([r[k] for r in rows]);valid=np.isfinite(a)&np.isfinite(b)
        fields[k]=dict(exact_equal=bool(np.array_equal(a,b,equal_nan=True)),
           max_abs_difference=float(np.max(abs(a[valid].astype(float)-b[valid].astype(float)))) if valid.any() else 0.)
    result=dict(passed=all(x['exact_equal'] for x in fields.values()),prefix_frames=n,full_frames=len(index),
       fields=fields,verified_black_frames=black,future_pixels_opened=False,
       new_stream=True,scope='sealed native online CSV plus matching current-flight pixels and supplied PX4 attitudes',
       native_not_recomputed=True,frozen_sources=frozen_manifest())
    writej(folder/'prefix-audit.json',result);print(json.dumps(result,indent=2),flush=True)
    assert result['passed']

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--build',action='store_true');a.add_argument('--smoke',type=Path);a.add_argument('--frames',type=int,default=200);a.add_argument('--verify',type=Path);v=a.parse_args()
    if v.build:print(build_native()[0])
    if v.smoke:smoke(v.smoke,v.frames)
    if v.verify:verify_stream(v.verify)

