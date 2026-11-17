#!/usr/bin/env bash
set -eo pipefail
root=/home/uav/paper_revision_20261117
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash
set -u
mkdir -p "$root/revision/data/qingyu_contract_probe" "$root/revision/generated/bin"
out="$root/revision/data/qingyu_contract_probe"
export ROS_MASTER_URI=http://127.0.0.1:11483
export ROS_HOSTNAME=127.0.0.1
export ROS_LOG_DIR="$out/ros_logs"
python3 - <<'PY'
import socket
s=socket.socket();s.bind(('127.0.0.1',11483));s.close()
PY
g++ -std=c++17 "$root/revision/scripts/qingyu_contract_probe.cpp" -I/opt/qingyu_api_ros1/include \
  -L/opt/qingyu_api_ros1/lib -Wl,-rpath,/opt/qingyu_api_ros1/lib -lqingyu_api_sdk \
  $(pkg-config --cflags --libs roscpp) -o "$root/revision/generated/bin/qingyu_contract_probe"
roscore -p 11483 >"$out/roscore.log" 2>&1 &
core_pid=$!
trap 'kill -INT "$core_pid" 2>/dev/null || true; wait "$core_pid" 2>/dev/null || true' EXIT
for i in $(seq 1 80); do
  if rosnode list >"$out/nodes_before.txt" 2>/dev/null; then break; fi
  sleep .1
done
PYTHONPATH="$root/mujoco_closed_loop:/home/uav/桌面/uav_arm_1/.mujoco_deps:$PYTHONPATH" python3 - <<'PY'
from pathlib import Path
import csv,math,json
from arm_command import ArmCommand
out=Path('/home/uav/paper_revision_20261117/revision/data/qingyu_contract_probe')
cases=[(0,90,True),(30,0,True),(60,90,True),(90,180,True),(118,-90,True),(120,450,True),(90,45,False)]
fields=[ArmCommand(i,i*.1,math.radians(t),math.radians(p),e).qingyu_joint_fields() for i,(t,p,e) in enumerate(cases)]
with (out/'input.csv').open('w') as f:
 w=csv.writer(f);w.writerow(fields[0].keys());w.writerows([[int(x) if isinstance(x,bool) else x for x in row.values()] for row in fields])
(out/'requested.json').write_text(json.dumps(cases,indent=2))
PY
timeout 20 "$root/revision/generated/bin/qingyu_contract_probe" "$out/input.csv" "$out/received.csv" | tee "$out/probe.txt"
rosnode list >"$out/nodes_after.txt"
python3 - <<'PY'
from pathlib import Path
import csv,json,hashlib
p=Path('/home/uav/paper_revision_20261117/revision/data/qingyu_contract_probe')
a=list(csv.DictReader((p/'input.csv').open()));b=list(csv.DictReader((p/'received.csv').open()))
assert len(a)==len(b)==7
for x,y in zip(a,b):
 for k,v in x.items():assert abs(float(v)-float(y[k]))<1e-4,(k,x,y)
for name in ('nodes_before.txt','nodes_after.txt'):
 nodes=(p/name).read_text().strip().splitlines()
 assert all(n=='/rosout' for n in nodes),nodes
result=dict(passed=True,commands_checked=len(a),transport='installed Qingyu C++ SDK to isolated ROS topic',
 master='http://127.0.0.1:11483',topic='/paper_revision_probe/command',serial_backend_started=False,
 sdk_library_sha256=hashlib.sha256(Path('/opt/qingyu_api_ros1/lib/libqingyu_api_sdk.so').read_bytes()).hexdigest(),
 verified=['degree payload preserved','single-section zeroing','118 degree cap','periodic phi','enable field preserved'],
 physical_feedback_verified=False,physical_angle_unit_verified=False,
 limitation='ROS message transport check; firmware interpretation and physical response require separate verification')
(p/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
PY
