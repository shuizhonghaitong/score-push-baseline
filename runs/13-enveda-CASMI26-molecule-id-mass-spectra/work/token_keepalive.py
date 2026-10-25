import datetime, os, subprocess, time
RUN = '/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra'
os.chdir(RUN)
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ[k] = 'http://172.28.4.16:18999'
PYK = '/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python'
SKILL = '/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py'
TOK = '/mnt/disk3/liuyutong_home/workspace/skill/score-push/.private/kaggle-access-token'
LOG = 'work/token_keepalive.log'
END = datetime.datetime(2026, 12, 14, 0, 0)
def log(msg):
    print(datetime.datetime.now().strftime('%F %T'), msg, file=open(LOG, 'a'), flush=True)

log('token_keepalive v2 start (slot every 8h from 11:45 10-23; first fires immediately)')
next_due = datetime.datetime(2026, 10, 23, 11, 45)
while datetime.datetime.now() < END:
    if os.path.exists('work/stop_keepalive.txt'):
        log('stop flag present; exiting'); break
    now = datetime.datetime.now()
    if now >= next_due:
        fired = next_due
        ok = False
        for attempt in range(18):
            r = subprocess.run([PYK, SKILL, 'auth-session', '--output', TOK, '--yes'],
                               capture_output=True, text=True, timeout=900)
            if r.returncode == 0:
                log('refresh ok (attempt %d) slot=%s' % (attempt + 1, fired.isoformat()))
                ok = True
                break
            log('refresh fail attempt %d rc=%s %s' % (attempt + 1, r.returncode,
                (r.stdout + r.stderr)[-250:].replace('\n', ' | ')))
            time.sleep(600)
        if not ok:
            log('ALARM refresh failed 3h straight at slot %s; retry next slot' % fired.isoformat())
        next_due = fired + datetime.timedelta(seconds=8 * 3600)
        time.sleep(120)
    time.sleep(120)
log('token_keepalive done')
