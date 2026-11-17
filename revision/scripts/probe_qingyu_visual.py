"""Read the installed SDK's visualization output on an isolated ROS master."""
from pathlib import Path
import json,time,hashlib
import rospy,rosnode
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker,MarkerArray
from qingyu_api.msg import QingyuCommand

ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'revision/data/qingyu_visual_probe';out.mkdir(parents=True,exist_ok=True)
rospy.init_node('paper_visual_probe',anonymous=False)
deadline=time.monotonic()+12
topic=None
while time.monotonic()<deadline:
    matches=[(n,t) for n,t in rospy.get_published_topics() if t in ('visualization_msgs/Marker','visualization_msgs/MarkerArray')]
    if matches:topic,kind=matches[0];break
    time.sleep(.1)
if topic is None:raise RuntimeError('SDK did not publish a marker topic')
pub=rospy.Publisher('/continuum_arm/control_angles',Float32MultiArray,queue_size=1)
commands=[]
sub=rospy.Subscriber('/paper_visual_probe/command',QingyuCommand,lambda m:commands.append(dict(theta1=m.theta1,phi1=m.phi1,cmd_mode=m.cmd_mode)),queue_size=50)
deadline=time.monotonic()+5
while pub.get_num_connections()==0 and time.monotonic()<deadline:time.sleep(.1)
def marker_dict(m):
    return dict(id=m.id,ns=m.ns,type=m.type,frame=m.header.frame_id,mesh=m.mesh_resource,
        scale=[m.scale.x,m.scale.y,m.scale.z],color=[m.color.r,m.color.g,m.color.b,m.color.a],
        position=[m.pose.position.x,m.pose.position.y,m.pose.position.z],
        quaternion=[m.pose.orientation.x,m.pose.orientation.y,m.pose.orientation.z,m.pose.orientation.w],
        points=[[p.x,p.y,p.z] for p in m.points])
cases=[]
for theta,phi in [(0,0),(90,0),(90,90),(90,180),(90,-90)]:
    commands.clear()
    for _ in range(4):pub.publish(Float32MultiArray(data=[theta,phi]));time.sleep(.1)
    message=rospy.wait_for_message(topic,MarkerArray if kind.endswith('MarkerArray') else Marker,timeout=3)
    markers=message.markers if hasattr(message,'markers') else [message]
    cases.append(dict(theta_deg=theta,phi_deg=phi,markers=[marker_dict(m) for m in markers],bridge_commands=commands[-3:]))
api=rospy.Publisher('/paper_visual_probe/command',QingyuCommand,queue_size=1)
angle_messages=[]
angle_sub=rospy.Subscriber('/continuum_arm/control_angles',Float32MultiArray,lambda m:angle_messages.append(list(m.data)),queue_size=30)
time.sleep(.2)
api_cases=[]
for phi in [0,90,180,-90]:
    angle_messages.clear()
    msg=QingyuCommand();msg.theta1=90;msg.phi1=phi;msg.enable=True;msg.cmd_mode=1;msg.control_mode=0
    for _ in range(3):api.publish(msg);time.sleep(.1)
    message=rospy.wait_for_message(topic,MarkerArray,timeout=3)
    api_cases.append(dict(theta1_deg=90,phi1_deg=phi,control_angles=angle_messages[-1] if angle_messages else None,
        end_position=[message.markers[-1].pose.position.x,message.markers[-1].pose.position.y,message.markers[-1].pose.position.z]))
nodes=rosnode.get_node_names()
assert all(n in ['/rosout','/continuum_arm_visualize','/paper_visual_probe','/qingyu_api_bridge'] for n in nodes),nodes
result=dict(source='/opt/qingyu_api_ros1/lib/continuum_arm/continuum_arm_visualize',
    binary_sha256=hashlib.sha256(Path('/opt/qingyu_api_ros1/lib/continuum_arm/continuum_arm_visualize').read_bytes()).hexdigest(),
    topic=topic,kind=kind,master=rospy.get_master().getUri()[2],nodes=nodes,
    hardware_nodes_started=False,cases=cases,api_cases=api_cases)
(out/'markers.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(topic=topic,kind=kind,api_cases=api_cases),indent=2))
