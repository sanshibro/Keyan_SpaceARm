"""Closed-loop shaping and acceleration trim for an offset centre of mass (ENU/FLU)."""
import numpy as np

FRD_FROM_FLU = np.diag([1., -1., -1.])
GRAVITY_M_S2 = 9.81
GEAR_TRIGGER = .008
GEAR_RELEASE = .011
THETA_MAX = np.deg2rad(120.)

def bound(v, maximum):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    return v * (maximum/norm) if norm > maximum else v.copy()

def shape_vector(theta, phi):
    return theta*np.array([np.cos(phi),np.sin(phi)])

class GravityTrim:
    """Map the estimated gravitational moment to the native acceleration command.

    The tilt perturbation uses the calibrated gain of PX4's cascaded attitude P,
    rate P, quad-X mixer and the simulator's force per actuator unit. A washout
    with the native rate integrator's P/I time constant transfers steady load to
    the flight controller. Physical rotor commands always come from PX4.
    """
    def __init__(self, unit_sign_torque, enabled, rotor_center_body=None):
        self.enabled=enabled
        self.rotor_center_body=np.zeros(3) if rotor_center_body is None else np.asarray(rotor_center_body,float)
        self.stiffness=np.asarray(unit_sign_torque,dtype=float)[:2,:2]*12*.707107*6.5*.15
        if not np.all(np.linalg.eigvalsh(self.stiffness)>0):raise ValueError('Invalid lateral allocation')
        self.integration_tau=np.array([.15/.2,.15/.4])
        self.integrated=None

    def evaluate(self, com_world, mass, orientation, dt,vertical_accel=0.):
        # Uniform gravity acts through the compound COM. The differential
        # rotor moment must cancel the real thrust line's offset from that COM.
        # At hover this equals the static gravity moment about the rotor hub.
        # A rigid vehicle with a COM directly below the rotors needs zero
        # torque trim as it pitches: a rotating world-gravity lever would
        # spuriously feed the position loop's attitude changes back into it.
        lever_body=orientation.T@com_world-self.rotor_center_body
        vertical_force=max(0.,GRAVITY_M_S2+vertical_accel)
        thrust=mass*vertical_force/max(orientation[2,2],.5)
        torque_flu=np.cross(lever_body,[0.,0.,thrust])
        required=(FRD_FROM_FLU@torque_flu)[:2]
        if self.integrated is None:self.integrated=required.copy()
        transient=required-self.integrated
        self.integrated+=(1.-np.exp(-dt/self.integration_tau))*transient
        tilt_frd=np.linalg.solve(self.stiffness,transient) if self.enabled else np.zeros(2)
        # A five-degree trim envelope leaves 40 degrees of the native 45-degree
        # flight envelope to the translational feedback and planned acceleration.
        tilt_frd=bound(tilt_frd,np.deg2rad(5.))
        rot_flu=FRD_FROM_FLU@np.r_[tilt_frd,0.]
        delta_up=orientation@np.cross(rot_flu,[0.,0.,1.])
        # Perturb horizontal thrust direction at constant vertical specific force.
        up=orientation[:,2];vertical=max(up[2],.5)
        horizontal=vertical_force*(delta_up[:2]/vertical-up[:2]*delta_up[2]/vertical**2)
        return np.r_[horizontal,0.],required,tilt_frd

def project_bend(estimate, desired, gap, evaluate):
    """Project a command onto a local clearance half-space and verify improvement."""
    if gap>=GEAR_TRIGGER:return desired.copy(), 'nominal'
    axes=np.eye(2)*np.deg2rad(.65)
    probe=[evaluate(bound(estimate+sign*ax,THETA_MAX)) for ax in axes for sign in (1.,-1.)]
    g=(np.array(probe[::2])-np.array(probe[1::2]))/(2*axes[0,0])
    magnitude=np.linalg.norm(g)
    if magnitude>1e-6:
        correction=max(0.,GEAR_RELEASE-gap-g@(desired-estimate))*g/magnitude**2
        candidate=bound(desired+correction,THETA_MAX)
        if evaluate(candidate)>=min(GEAR_RELEASE,gap+.0005):
            return candidate,'projected'
    # A min-distance field has a corner where opposing leg clearances tie.
    # Search directions in bending-vector space, retaining secondary constraints.
    options=[bound(desired,THETA_MAX),estimate.copy()]
    for radius in np.deg2rad([3.,8.,16.]):
        for phi in np.arange(8)*np.pi/4.:
            options.append(bound(estimate+radius*np.array([np.cos(phi),np.sin(phi)]),THETA_MAX))
    ranked=[(evaluate(b),-np.linalg.norm(b-desired),i) for i,b in enumerate(options)]
    best=max(ranked)
    viable=[r for r in ranked if r[0]>=GEAR_RELEASE]
    if viable:best=max(viable,key=lambda r:r[1])
    return options[best[2]],('projected' if best[0]>gap+.00003 else 'no_clearance_ascent')

class ShapeObservation:
    """Causal polar derivatives and ENU motion for online replanning."""
    def __init__(self,phi_start=np.pi):
        self.phi=phi_start;self.previous=None;self.dq=np.zeros(6);self.ddq=np.zeros(6)
    def update(self,t,xyz,velocity,yaw,b):
        theta=np.linalg.norm(b)
        if theta>=np.deg2rad(4.):
            angle=np.arctan2(b[1],b[0]);self.phi+=np.arctan2(np.sin(angle-self.phi),np.cos(angle-self.phi))
        q=np.r_[xyz,yaw,theta,self.phi]
        if self.previous is None or t-self.previous[0]>.15:
            self.dq=np.r_[velocity,np.zeros(3)];self.ddq=np.zeros(6)
        else:
            last_t,last_q=self.previous;dt=t-last_t
            if dt<=0:raise ValueError('Non-increasing flight clock')
            delta=q-last_q;delta[3]=np.arctan2(np.sin(delta[3]),np.cos(delta[3]))
            beta=1.-np.exp(-dt/.10)
            derivative=np.r_[velocity,delta[3:]/dt];prior=self.dq.copy()
            self.dq+=beta*(derivative-self.dq)
            self.ddq+=beta*((self.dq-prior)/dt-self.ddq)
        self.previous=(t,q.copy())
        return q,self.dq.copy(),self.ddq.copy()
