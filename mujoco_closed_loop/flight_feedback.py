"""Sensor-only inertial propagation and model-based rotor controller. No MuJoCo import."""
from collections import deque
import numpy as np,cv2

class InertialVisualState:
    def __init__(self,position):
        # Corrections older than 300 ms are rejected; 400 samples retain the
        # complete admissible history at 1 kHz without copying two seconds.
        self.p=np.array(position,dtype=float);self.v=np.zeros(3);self.R=np.eye(3);self.history=deque(maxlen=400)
        self.last_visual=0.;self.rejected=0;self.last_measurement=None
        self.position_measurements=deque(maxlen=30);self.last_velocity_measurement=np.zeros(3)
    def propagate(self,stamp,gyro,specific_force,dt):
        self.R=self.R@cv2.Rodrigues(np.asarray(gyro)*dt)[0]
        a=self.R@np.asarray(specific_force)+[0,0,-9.81]
        self.p+=self.v*dt+.5*a*dt*dt;self.v+=a*dt
        self.history.append((stamp,self.p.copy(),self.R.copy(),self.v.copy()))
    def correct(self,stamp,T_WU,now,velocity_gain=.15,velocity_window=.01,orientation_gain=.12,position_gain=.2,low_speed_gain=None,velocity_hint=None):
        if now-stamp>.3 or not self.history:self.rejected+=1;return False
        old=min(self.history,key=lambda x:abs(x[0]-stamp))
        innovation=T_WU[:3,3]-old[1]
        if np.linalg.norm(innovation)>.25:self.rejected+=1;return False
        shift=position_gain*innovation;self.p+=shift
        velocity_shift=np.zeros(3)
        previous=None
        for candidate in reversed(self.position_measurements):
            if stamp-candidate[0]>=velocity_window:
                previous=candidate;break
        if velocity_hint is not None and np.isfinite(velocity_hint).all() and np.linalg.norm(velocity_hint)<2:
            velocity=np.asarray(velocity_hint,dtype=float)
            gain=float(np.clip(velocity_gain,0,1))
            velocity_shift=gain*(velocity-old[3]);self.v+=velocity_shift
            self.last_velocity_measurement=velocity
        elif previous is not None:
            interval=stamp-previous[0]
            velocity=(T_WU[:3,3]-previous[1])/interval
            if np.linalg.norm(velocity)<2:
                gain=float(np.clip(low_speed_gain if low_speed_gain is not None and np.linalg.norm(velocity)<.08 else velocity_gain,0,1))
                velocity_shift=gain*(velocity-old[3]);self.v+=velocity_shift
                self.last_velocity_measurement=velocity
        self.last_measurement=(stamp,T_WU[:3,3].copy())
        self.position_measurements.append(self.last_measurement)
        self.history=deque(((t,p+shift,R,v+velocity_shift) for t,p,R,v in self.history),maxlen=2000)
        # Small delayed orientation residual applied in world coordinates.
        delta=T_WU[:3,:3]@old[2].T;rotation=cv2.Rodrigues(delta)[0].ravel()
        correction=cv2.Rodrigues(orientation_gain*rotation)[0]
        self.R=correction@self.R
        self.history=deque(((t,p,correction@R,v) for t,p,R,v in self.history),maxlen=2000)
        self.last_visual=stamp;return True

    def correct_stationary_velocity(self,now,gyro,specific_force,dt=.001):
        """Zero-velocity update from a stationary LiDAR position window and IMU."""
        measured_force=np.asarray(specific_force,dtype=float)
        if len(self.position_measurements)<6 or now-self.position_measurements[-1][0]>.2:return False
        latest=self.position_measurements[-1]
        older=next((item for item in reversed(self.position_measurements)
                    if latest[0]-item[0]>=.5),None)
        if older is None or np.linalg.norm(latest[1]-older[1])>.006:return False
        if np.linalg.norm(gyro)>.05 or np.linalg.norm(measured_force[:2])>.3:return False
        self.v*=np.exp(-20.*dt)
        return True

class RotorController:
    @classmethod
    def from_model(cls, model):
        from super_metric_model import allocation_matrix
        controller = cls(mass=float(model.body_mass.sum()))
        controller.inv = np.linalg.inv(allocation_matrix(model))
        return controller

    def __init__(self,mass=2.25):
        self.mass=mass;self.integral=np.zeros(3);self.velocity_integral=np.zeros(3)
        self.position_gain=2.;self.velocity_gain=2.2;self.integral_gain=.15
        self.integral_leak_s=None;self.horizontal_acceleration_limit=None
        xy=np.array([[.2,.2],[-.2,.2],[-.2,-.2],[.2,-.2]])
        self.inv=np.linalg.inv(np.array([np.ones(4),xy[:,1],-xy[:,0],[.02,-.02,.02,-.02]]))
    def command(self,state,gyro,target,com_body,dt=.001,yaw=0.,target_velocity=None,target_acceleration=None):
        error=np.asarray(target)-state.p
        decay=np.exp(-dt/self.integral_leak_s) if self.integral_leak_s else 1.
        self.integral=np.clip(decay*self.integral+error*dt,-.3,.3)
        acc=self.position_gain*error-self.velocity_gain*state.v+self.integral_gain*self.integral+[0,0,9.81]
        if target_velocity is not None:acc+=self.velocity_gain*np.asarray(target_velocity)
        if target_acceleration is not None:acc+=np.asarray(target_acceleration)
        if getattr(state,'visual_servo',False):
            desired_velocity=np.clip(.5*error,[-.06,-.06,-.02],[.06,.06,.02])
            velocity_error=desired_velocity-state.v
            self.velocity_integral=np.clip(self.velocity_integral+velocity_error*dt,-.5,.5)
            acc=1.5*velocity_error+.3*self.velocity_integral+np.array([0,0,9.81])
        if self.horizontal_acceleration_limit is not None:
            acc[:2]=np.clip(acc[:2],-self.horizontal_acceleration_limit,self.horizontal_acceleration_limit)
        z=acc/np.linalg.norm(acc);sy,cy=np.sin(yaw),np.cos(yaw)
        y=np.array([-z[2]*sy,z[2]*cy,z[0]*sy-z[1]*cy]);y/=np.linalg.norm(y)
        x=np.array([y[1]*z[2]-y[2]*z[1],y[2]*z[0]-y[0]*z[2],y[0]*z[1]-y[1]*z[0]])
        Rd=np.column_stack([x,y,z])
        M=Rd.T@state.R-state.R.T@Rd;er=.5*np.array([M[2,1],M[0,2],M[1,0]])
        g=state.R.T@np.array([0,0,-self.mass*9.81]);c=com_body
        gravity=np.array([c[1]*g[2]-c[2]*g[1],c[2]*g[0]-c[0]*g[2],c[0]*g[1]-c[1]*g[0]])
        torque=-8*er-1.6*np.asarray(gyro)-gravity
        raw=self.inv@np.r_[max(0,self.mass*np.dot(acc,state.R[:,2])),torque]
        return np.clip(raw,0,12),bool(np.any((raw<0)|(raw>12)))
