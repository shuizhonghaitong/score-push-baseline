import datetime, os, subprocess, time
RUN = '/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra'
os.chdir(RUN)
KAGGLE = '/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle'
PYK = f'{KAGGLE}/bin/python'
SKILL = '/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py'
TOK = '/mnt/disk3/liuyutong_home/workspace/skill/score-push/.private/kaggle-access-token'
LOG = 'work/track_b2.log'
def log(msg):
    print(datetime.datetime.now().strftime('%F %T'), msg, file=open(LOG, 'a'), flush=True)

def refresh_token():
    r = subprocess.run([PYK, SKILL, 'auth-session', '--output', TOK, '--yes'], capture_output=True, text=True)
    log('token refresh rc=' + str(r.returncode) + ' ' + r.stdout[-200:].replace('\n', ' | '))
    return r.returncode == 0

log('track_b2 start; waiting for nb_sub_pushed.txt')
while not os.path.exists('work/nb_sub_pushed.txt'):
    if datetime.datetime.now() > datetime.datetime(2026, 10, 24, 12, 0):
        log('no nb_sub push flag by deadline; giving up')
        raise SystemExit(3)
    time.sleep(120)
log('nb_sub push flag seen')

r = subprocess.run([PYK, SKILL, 'notebook-wait', 'lilac1996/casmi26-v2-entropy',
                    '--out', 'downloads/nb_sub2', '--timeout', '28800', '--interval', '120'],
                   capture_output=True, text=True)
log(f'notebook-wait rc={r.returncode} {r.stdout[-200:]} {r.stderr[-150:]}')

sub = None
for root, _, files in os.walk('downloads/nb_sub2'):
    for f in files:
        if f == 'submission.csv':
            sub = os.path.join(root, f)
ok = False
if sub and os.path.exists(sub):
    import pandas as pd
    s = pd.read_csv(sub)
    ok = (list(s.columns) == ['molecule_id', 'smiles'] and len(s) == 400
          and s.smiles.notna().all() and (s.smiles.astype(str).str.len() > 0).all())
    real = sum(1 for x in s.smiles if str(x) != 'CCO')
    log(f'smoke rows={len(s)} ok={ok} non-CCO={real}')
    if real < 380:
        log('quality floor failed (non-CCO < 380); aborting track B2')
        raise SystemExit(4)
if not ok:
    log('smoke not verified; aborting track B2')
    raise SystemExit(2)
open('work/nb_sub_ready.txt', 'w').write(datetime.datetime.now().isoformat() + '\n')

if (time.time() - os.path.getmtime(TOK)) > 8 * 3600:
    refresh_token()
r = subprocess.run([f'{KAGGLE}/bin/kaggle', 'competitions', 'submit',
                    'enveda-CASMI26-molecule-id-mass-spectra',
                    '-k', 'lilac1996/casmi26-v2-entropy',
                    '-m', 'backup entropy/JS spectral engine (autonomous run 13, track B2)'],
                   capture_output=True, text=True, timeout=600)
log('submit rc=' + str(r.returncode) + ' ' + r.stdout[-300:] + ' ' + r.stderr[-200:])
if r.returncode == 0:
    open('work/nb_sub_submitted.txt', 'w').write(datetime.datetime.now().isoformat() + '\n' + r.stdout[-500:] + '\n')

for i in range(120):
    r = subprocess.run([f'{KAGGLE}/bin/kaggle', 'competitions', 'submissions',
                        'enveda-CASMI26-molecule-id-mass-spectra'], capture_output=True, text=True, timeout=120)
    out = r.stdout
    if 'IN PROGRESS' not in out and 'INPROGRESS' not in out.upper():
        log('terminal:\n' + out[:2000])
        open('work/nb_sub_score.txt', 'w').write(out[:3000] + '\n')
        break
    time.sleep(600)
log('track_b2 done')
