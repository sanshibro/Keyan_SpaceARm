"""Inventory local reference headers and SDK documents. Uses filesystem access only."""
import argparse,csv,hashlib,json
from pathlib import Path
import re
import subprocess

def inspect(base,sdk):
    families=['angle_runs','orientation_fai_runs','clean_fai_runs','fai_runs','fai_position_runs',
        'orb_strain_runs','orb_theta_runs','orb_fai_runs','live_strain_theta_runs','live_dual_strain_theta_fai_runs',
        'paired_orientation_fai_runs','dual_strain_theta_fai_models','dual_strain_ai','Result']
    mass_key=re.compile(r'(payload|ballast|added.?mass|added.?weight|test.?mass|hanging.?mass|砝码|载荷|挂重|负载质量)',re.I)
    group={}
    for folder in families:
        files=sorted((base/folder).rglob('*.csv'))
        if not files:continue
        seen=set();dose=[];count=0
        for file in files:
            with file.open(encoding='utf-8-sig',errors='replace') as f:cols=next(csv.reader(f),[])
            if not cols:continue
            count+=1;seen.update(cols)
            if any(mass_key.search(c) for c in cols):dose.append(file.relative_to(base).as_posix())
        group[folder]=dict(tables=count,payload_column_candidates=dose,
            signal_examples=sorted(c for c in seen if re.search(r'time|theta_deg|orb_theta|theta_true|cmd_theta|strain_value|mocap',c,re.I))[:18])
    paths=['include/qingyu_api_sdk/client.hpp','share/qingyu_api_sdk/examples/cli.cpp',
        'share/qingyu_api_sdk/README.md','share/qingyu_api_sdk/USAGE.md','share/qingyu_api/srv/ConfigReadOne.srv',
        'share/qingyu_api/srv/ConfigWriteOne.srv','share/continuum_arm/config/mcu_packet_sender.yaml','lib/continuum_arm/continuum_arm_slider_gui.py']
    hashes={};matches=[]
    for s in paths:
        f=sdk/s
        if not f.is_file():continue
        b=f.read_bytes();hashes[s]=hashlib.sha256(b).hexdigest()
        for n,line in enumerate(b.decode('utf8',errors='replace').splitlines(),1):
            if re.search(r'flexibility.?ratio|stiffness.?ratio|compliance.?ratio|柔性比|柔顺度|刚度系数|刚度比|软硬度',line,re.I):matches.append({'file':s,'line':n})
    version=subprocess.run(['dpkg-query','-W','-f=${Version}','qingyu-api-ros1-a01'],capture_output=True,text=True)
    return dict(sdk_package='qingyu-api-ros1-a01',version=version.stdout.strip(),source_sha256=hashes,named_ratio_matches=matches,
        families=group,total_nonempty_csv=sum(v['tables'] for v in group.values()),
        payload_column_candidates=sum(len(v['payload_column_candidates']) for v in group.values()),
        scope='Header audit of listed local bench folders and text scan of listed installed SDK files. Payloads may also have been recorded out of band. No hardware connection or configuration changes.',
        register_semantics='Generic typed configuration/readback is documented. Specific ratio ID, unit, type and calibration are not established from this audit.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    v=inspect(a.base,Path('/opt/qingyu_api_ros1'));a.output.write_text(json.dumps(v,indent=2)+'\n')
    print(v['version'],v['total_nonempty_csv'],v['payload_column_candidates'],v['named_ratio_matches'])
