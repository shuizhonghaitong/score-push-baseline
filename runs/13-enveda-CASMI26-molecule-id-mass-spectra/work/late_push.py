import datetime, os, subprocess, time
RUN = '/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra'
os.chdir(RUN)
KAGGLE = '/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle'
PYK = f'{KAGGLE}/bin/python'
SKILL = '/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py'
TOK = '/mnt/disk3/liuyutong_home/workspace/skill/score-push/.private/kaggle-access-token'
LOG = 'work/late_push.log'
def log(msg):
    print(datetime.datetime.now().strftime('%F %T'), msg, file=open(LOG, 'a'), flush=True)

def refresh_token():
    r = subprocess.run([PYK, SKILL, 'auth-session', '--output', TOK, '--yes'], capture_output=True, text=True)
    log('token refresh rc=' + str(r.returncode) + ' ' + r.stdout[-200:].replace('\n', ' | '))
    return r.returncode == 0

def token_fresh():
    try:
        return (time.time() - os.path.getmtime(TOK)) < 8 * 3600
    except Exception:
        return False

def push_once(folder):
    if not token_fresh():
        refresh_token()
    import kaggle
    try:
        kaggle.api.kernels_push(folder)
        return 'ok'
    except Exception as e:
        resp = getattr(e, 'response', None)
        code = getattr(resp, 'status_code', None) if resp is not None else None
        if code == 429:
            ra = int(resp.headers.get('retry-after', '45000'))
            log(f'{folder} 429 retry_after={ra}s (~{ra//3600}h{ra%3600//60}m)')
            return f'429:{ra}'
        log(f'{folder} PUSH_ERR code={code} {repr(e)[:200]}')
        return 'err'

def push_with_retry(folder, flag, rounds=8):
    for i in range(rounds):
        r = push_once(folder)
        if r == 'ok':
            open(flag, 'w').write(datetime.datetime.now().isoformat() + '\n')
            log(f'{folder} PUSH_OK -> {flag}')
            return True
        if r.startswith('429'):
            wait = min(int(r.split(':')[1]) + 120, 24 * 3600)
            log(f'waiting {wait}s before round {i+2}')
            time.sleep(wait)
        else:
            if not token_fresh():
                refresh_token()
            time.sleep(900)
    log(f'{folder} push failed all {rounds} rounds')
    return False

log('late_push start (recovery after 02:50 401 + 429 window)')
t_refresh = datetime.datetime(2026, 10, 23, 15, 10)
t_v465 = datetime.datetime(2026, 10, 23, 15, 30)
t_sub = datetime.datetime(2026, 10, 23, 15, 40)
while datetime.datetime.now() < t_refresh:
    time.sleep(300)
refresh_token()
while datetime.datetime.now() < t_v465:
    time.sleep(60)
push_with_retry('work/nb_v465', 'work/v465_pushed.txt')
while datetime.datetime.now() < t_sub:
    time.sleep(60)
push_with_retry('work/nb_sub', 'work/nb_sub_pushed.txt')
log('late_push done')
