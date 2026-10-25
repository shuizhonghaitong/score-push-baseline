import subprocess, time, datetime, os, json, re
RUN = '/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra'
os.chdir(RUN)
KAGGLE = '/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin'
LOG = 'work/submit_gate.log'
def log(msg):
    print(datetime.datetime.now().strftime('%F %T'), msg, file=open(LOG, 'a'), flush=True)

def quota_gpu_remaining():
    r = subprocess.run([f'{KAGGLE}/kaggle', 'quota', '--format', 'json'], capture_output=True, text=True, timeout=120)
    try:
        data = json.loads(r.stdout)
        for row in data if isinstance(data, list) else data.get('rows', []):
            if row.get('resource') == 'GPU' or 'gpu' in str(row.get('resource', '')).lower():
                return float(str(row.get('remaining', '0h')).replace('h', ''))
    except Exception as e:
        log('quota parse err ' + repr(e) + ' raw=' + r.stdout[:200])
    return None

def smoke_ok():
    try:
        txt = open('work/auto_v465.log').read()
        m = re.findall(r'READY_TO_SUBMIT VERSION=\S* SUB=(\S*) verify_rc=(\d+)', txt)
        if m:
            sub, rc = m[-1]
            if rc == '0' and sub and os.path.exists(sub):
                return True
        seq = open('work/sequencer.log').read() if os.path.exists('work/sequencer.log') else ''
        if 'ok=True' in seq:
            return True
    except Exception:
        pass
    return False

log('submit_gate start; gate: quota reset 10-24 08:00 CST + gpu>=3h + smoke verified')
gate_from = datetime.datetime(2026, 10, 24, 8, 35)
while datetime.datetime.now() < gate_from:
    time.sleep(300)
for attempt in range(40):
    now = datetime.datetime.now()
    q = quota_gpu_remaining()
    s = smoke_ok()
    log(f'check {now}: gpu_remaining={q} smoke_ok={s}')
    if q is not None and q >= 3.0 and s:
        log('GATE OPEN -> submitting v465 fork (latest version)')
        r = subprocess.run([f'{KAGGLE}/kaggle', 'competitions', 'submit',
                            'enveda-CASMI26-molecule-id-mass-spectra',
                            '-k', 'lilac1996/casmi26-v465-fork',
                            '-m', 'v465 fork v1: V15 lineage unchanged + sim-score dump (autonomous run 13)'],
                           capture_output=True, text=True, timeout=600)
        log('submit rc=' + str(r.returncode) + ' out=' + r.stdout[-500:] + ' err=' + r.stderr[-300:])
        if r.returncode == 0:
            open('work/v465_submitted.txt', 'w').write(now.isoformat() + '\n' + r.stdout[-800:] + '\n')
            log('SUBMITTED; starting watch')
            break
        time.sleep(1800)
    else:
        time.sleep(900)
# watch the scored rerun
for i in range(240):
    r = subprocess.run([f'{KAGGLE}/kaggle', 'competitions', 'submissions',
                        'enveda-CASMI26-molecule-id-mass-spectra'], capture_output=True, text=True, timeout=120)
    out = r.stdout
    if 'IN PROGRESS' not in out and 'INPROGRESS' not in out.upper():
        log('submission terminal state:\n' + out[:1500])
        open('work/v465_score.txt', 'w').write(out[:3000] + '\n')
        break
    time.sleep(600)
log('submit_gate done')
