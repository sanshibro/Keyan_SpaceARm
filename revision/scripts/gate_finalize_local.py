"""Finalize the two-page review after all registered raw flights and camera panes are verified."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from pypdf import PdfReader


def ready(stage):
    state=stage/'records/verified-evidence.json'
    try:
        return (state.is_file() and json.loads(state.read_bytes())['count']==60 and
                (stage/'review/video-verification.json').is_file())
    except (OSError,ValueError):
        return False


def finalize(stage):
    source=Path(__file__).resolve().parent
    rd=stage/'review';qa=stage/'qa/final';qa.mkdir(exist_ok=True,parents=True)
    for cmd in ([source/'gate_cohort.py',stage],[source/'gate_paired_outcomes.py',stage],
                [source/'gatereview_pdf.py','--root',stage]):
        subprocess.run([sys.executable,*map(str,cmd)],check=True)
    pdf=rd/'gate_review.pdf'
    r=PdfReader(str(pdf))
    if len(r.pages)!=2:raise ValueError('Review must contain exactly two pages')
    body='\n'.join(p.extract_text() for p in r.pages)
    if '60/60' not in body or 'Gate passage' not in body:
        raise ValueError('Formal score and vector figure text missing from PDF')
    with (rd/'all_trials_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        table=list(csv.DictReader(f))
    if len({v['case'] for v in table})!=60:raise ValueError('Scored trial table incomplete')
    proof=json.loads((rd/'video-verification.json').read_bytes())
    if proof['verified_wrist_tiles']!=5*proof['frames']:raise ValueError('Video camera verification incomplete')
    subprocess.run(['pdftoppm','-q','-r','144','-png',str(pdf),str(qa/'review')],check=True)
    for i in (1,2):
        if not (qa/f'review-{i}.png').is_file():raise IOError('Missing rendered review page')
    outputs=[pdf,rd/'gate_keyframes.pdf',rd/'gate_four_metrics.pdf',rd/'gate_method_details.pdf',
             rd/'gate_paired_outcomes.pdf', rd/'gate_multiview_comparison.mp4',rd/'all_trials_metrics.csv']
    proof={'pages':2,'cases':60,'report_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
           'media':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}}
    (qa/'release-ready.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print('Two final review pages, 60 original flights and five-channel video verified.',flush=True)


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('stage',type=Path);q.add_argument('--watch',action='store_true')
    args=q.parse_args();stage=args.stage.resolve()
    while not ready(stage):
        if not args.watch:raise SystemExit('Waiting for all sixty records and camera verification')
        time.sleep(45)
    finalize(stage)

if __name__=='__main__':
    main()
