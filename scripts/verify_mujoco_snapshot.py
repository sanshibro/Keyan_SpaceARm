#!/usr/bin/env python3
"""Verify copied source and development evidence against the export manifest."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    manifest = json.loads((ROOT / 'provenance/mujoco_publication_manifest.json').read_text())
    # Sole intentional modification of an exported source: portable ROS config lookup.
    adjustments = {'mujoco_closed_loop/ros_ws/launch/mujoco_fast_lio.launch':
                   'b54df116bf09a78a97f8be08bbd248be9d0bdb84bfd5556da545a63198a73bb8'}
    failures = []
    for item in manifest['source_files']:
        path = ROOT / item['path']
        if not path.is_file() or sha(path) != adjustments.get(item['path'], item['sha256']):
            failures.append(item['path'])
    if (ROOT / '.git').exists():
        tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=str(ROOT)).decode().split('\0'))
        for item in manifest['source_files']:
            if item['path'] not in tracked:
                failures.append('not in Git index: ' + item['path'])
    source_count = 0
    for path in (ROOT / 'mujoco_closed_loop').glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
        source_count += 1
    for path in (ROOT / 'mujoco_closed_loop').glob('*.sh'):
        subprocess.run(['bash', '-n', str(path)], check=True)
    run = ROOT / 'mujoco_closed_loop/runs' / manifest['latest_run'] / 'manifest.json'
    result = json.loads(run.read_text())
    if result['passed'] is not False or result['navigation_completed'] is not False:
        failures.append('latest development outcome unexpectedly changed')
    print(json.dumps(dict(passed=not failures, copied_files=len(manifest['source_files']),
                         python_modules_parsed=source_count, failures=failures,
                         latest_full_task_accepted=result['full_task_accepted']), indent=2))
    raise SystemExit(bool(failures))

if __name__ == '__main__':
    main()
