import kaggle, time, datetime, os, subprocess
RUN = '/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra'
os.chdir(RUN)
KAGGLE = '/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin'
PYK = f'{KAGGLE}/python'
SKILL = '/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py'
def log(msg):
    print(datetime.datetime.now().strftime('%F %T'), msg, file=open('work/track_b.log', 'a'), flush=True)

log('track_b start; waiting until 10-23 03:30')
target = datetime.datetime(2026, 10, 23, 3, 30)
while datetime.datetime.now() < target:
    time.sleep(300)

if os.path.exists('work/nb_sub_validated.txt'):
    log('local validation present')
else:
    log('NOTE: no local validation yet; proceeding on smoke-based validation (task allows iterative submissions)')

pushed = os.path.exists('work/nb_sub_pushed.txt')
if pushed:
    log('nb_sub already pushed by sequencer; joining at smoke wait')
else:
    for i in range(5):
        try:
            kaggle.api.kernels_push('work/nb_sub')
            log(f'NB_SUB_PUSH_OK round {i}')
            pushed = True
            open('work/nb_sub_pushed.txt', 'w').write(datetime.datetime.now().isoformat() + '\n')
            break
        except Exception as e:
            resp = getattr(e, 'response', None)
            if resp is not None and resp.status_code == 429:
                ra = int(resp.headers.get('retry-after', '1800'))
                log(f'NB_SUB 429 retry_after={ra} round {i}')
                time.sleep(min(ra + 120, 5 * 3600))
            else:
                log('NB_SUB_ERR ' + repr(e)[:200])
                time.sleep(1200)
    if not pushed:
        log('NB_SUB push failed all rounds; aborting track B')
        raise SystemExit(1)

r = subprocess.run([PYK, SKILL, 'notebook-wait', 'lilac1996/casmi26-v2-entropy',
                    '--out', 'downloads/nb_sub', '--timeout', '14400', '--interval', '120'],
                   capture_output=True, text=True)
log(f'notebook-wait rc={r.returncode} {r.stdout[-200:]} {r.stderr[-150:]}')

sub = None
for root, _, files in os.walk('downloads/nb_sub'):
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
        log(f'quality floor failed (non-CCO {real} < 380); aborting track B (no low-quality submit)')
        raise SystemExit(4)
if not ok:
    log('smoke not verified; aborting track B')
    raise SystemExit(2)
open('work/nb_sub_ready.txt', 'w').write(datetime.datetime.now().isoformat() + '\n')

r = subprocess.run([f'{KAGGLE}/kaggle', 'competitions', 'submit',
                    'enveda-CASMI26-molecule-id-mass-spectra',
                    '-k', 'lilac1996/casmi26-v2-entropy',
                    '-m', 'backup entropy/JS spectral engine (autonomous run 13, track B)'],
                   capture_output=True, text=True, timeout=600)
log('submit rc=' + str(r.returncode) + ' ' + r.stdout[-300:] + ' ' + r.stderr[-200:])
if r.returncode == 0:
    open('work/nb_sub_submitted.txt', 'w').write(datetime.datetime.now().isoformat() + '\n' + r.stdout[-500:] + '\n')

for i in range(120):
    r = subprocess.run([f'{KAGGLE}/kaggle', 'competitions', 'submissions',
                        'enveda-CASMI26-molecule-id-mass-spectra'], capture_output=True, text=True, timeout=120)
    out = r.stdout
    if 'IN PROGRESS' not in out and 'INPROGRESS' not in out.upper():
        log('terminal:\n' + out[:2000])
        open('work/nb_sub_score.txt', 'w').write(out[:3000] + '\n')
        break
    time.sleep(600)
log('track_b done')
