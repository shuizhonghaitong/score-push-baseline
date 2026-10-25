import datetime, glob, os, subprocess, time
RUN = '/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra'
os.chdir(RUN)
LOG = 'work/v465_watch2.log'
def log(msg):
    print(datetime.datetime.now().strftime('%F %T'), msg, file=open(LOG, 'a'), flush=True)

start = datetime.datetime(2026, 10, 24, 8, 15)
end = datetime.datetime(2026, 10, 24, 20, 0)
log(f'v465_watch2 start; window {start} -> {end}')
while datetime.datetime.now() < start:
    time.sleep(300)
seen = None
while datetime.datetime.now() < end:
    if os.path.exists('work/v465_submitted.txt'):
        log('v465 already submitted; exiting'); break
    r = subprocess.run(['/usr/bin/timeout', '600', 'kaggle', 'kernels', 'output', 'lilac1996/casmi26-v465-fork', '-p', 'downloads/v465_w2',
                        '--file-pattern', 'submission.csv'], capture_output=True, text=True)
    sub = None
    for root, _, files in os.walk('downloads/v465_w2'):
        for f in files:
            if f == 'submission.csv':
                p = os.path.join(root, f)
                if sub is None or os.path.getmtime(p) > os.path.getmtime(sub):
                    sub = p
    if sub and os.path.exists(sub):
        if seen == sub and os.path.getsize(sub) == seen[1]:
            pass
        rc = subprocess.run(['./.venv/bin/python', '-c',
            'import pandas as pd,sys\n'
            's=pd.read_csv(sys.argv[1])\n'
            'assert list(s.columns)==["molecule_id","smiles"]\n'
            'assert len(s)==400, len(s)\n'
            'assert s.smiles.notna().all()\n'
            'n=sum(1 for x in s.smiles if str(x)!="CCO")\n'
            'print("rows=400 non-CCO=",n)\n'
            'sys.exit(0 if n>=350 else 1)', sub], capture_output=True, text=True)
        if rc.returncode == 0:
            line = f'READY_TO_SUBMIT VERSION=watch2 SUB={sub} verify_rc=0 {datetime.datetime.now().isoformat()}'
            with open('work/auto_v465.log', 'a') as f:
                f.write(line + '\n')
            open('work/v465_smoke_ok.txt', 'w').write(sub + '\n')
            log('SMOKE VERIFIED -> appended READY_TO_SUBMIT to auto_v465.log: ' + rc.stdout.strip())
            break
        log('output present but verify failed: ' + rc.stdout[-200:] + ' ' + rc.stderr[-200:])
    else:
        log('output poll rc=' + str(r.returncode) + ' ' + (r.stdout + r.stderr)[-150:].replace('\n', ' | '))
    if seen is None and sub:
        seen = (sub, os.path.getsize(sub))
    time.sleep(600)
log('v465_watch2 done')
