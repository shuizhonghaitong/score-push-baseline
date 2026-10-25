import kaggle, time, datetime, os, subprocess
RUN = '/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra'
os.chdir(RUN)
LOG = 'work/sequencer.log'
def log(msg):
    print(datetime.datetime.now().strftime('%F %T'), msg, file=open(LOG, 'a'), flush=True)

# 1) wait for the v465 push flag (night_push owns the 429 window)
log('sequencer start; waiting for v465_pushed.txt')
while not os.path.exists('work/v465_pushed.txt'):
    if datetime.datetime.now() > datetime.datetime(2026, 10, 24, 23, 0):
        log('GAVE UP waiting for push flag'); break
    time.sleep(300)
else:
    log('v465 push flag seen')

    # 2) wait for the v465 smoke to finish
    log('waiting for v465 smoke run')
    r = subprocess.run(['/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python',
                        '/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py',
                        'notebook-wait', 'lilac1996/casmi26-v465-fork', '--out', 'downloads/v465_seq',
                        '--timeout', '21600', '--interval', '120'], capture_output=True, text=True)
    log(f'notebook-wait rc={r.returncode} {r.stdout[-300:]} {r.stderr[-200:]}')

    # 3) verify smoke output
    sub = None
    for root, _, files in os.walk('downloads'):
        for f in files:
            if f == 'submission.csv':
                sub = os.path.join(root, f)
    if sub and os.path.exists(sub):
        import pandas as pd
        s = pd.read_csv(sub)
        ok = (list(s.columns) == ['molecule_id', 'smiles'] and len(s) == 400
              and s.smiles.notna().all() and (s.smiles.astype(str).str.len() > 0).all())
        log(f'smoke submission rows={len(s)} ok={ok} file={sub}')
    else:
        log('smoke submission.csv NOT FOUND')

    # 4) push the backup engine (CPU notebook) now that the 429 window is known-clear
    for i in range(3):
        try:
            kaggle.api.kernels_push('work/nb_sub')
            log('NB_SUB_PUSH_OK')
            open('work/nb_sub_pushed.txt', 'w').write(datetime.datetime.now().isoformat() + '\n')
            break
        except Exception as e:
            resp = getattr(e, 'response', None)
            if resp is not None and resp.status_code == 429:
                ra = int(resp.headers.get('retry-after', '1800'))
                log(f'NB_SUB 429 retry_after={ra}')
                time.sleep(min(ra + 120, 6 * 3600))
            else:
                log('NB_SUB_ERR ' + repr(e)[:200]); time.sleep(900)
log('sequencer done')
