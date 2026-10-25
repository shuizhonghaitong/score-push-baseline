import datetime, os, time, json
LOG = 'work/labels.log'
def log(msg):
    print(datetime.datetime.now().strftime('%F %T'), msg, file=open(LOG, 'a'), flush=True)

def wait_flag(flags, deadline):
    while datetime.datetime.now() < deadline:
        for f in flags:
            if os.path.exists(f):
                return f
        time.sleep(120)
    return None

def push_once():
    import kaggle
    try:
        kaggle.api.kernels_push('work/nb_labels')
        log('LABELS_PUSH_OK')
        return 'ok'
    except Exception as e:
        resp = getattr(e, 'response', None)
        if resp is not None and resp.status_code == 429:
            ra = int(resp.headers.get('retry-after', '3600'))
            log(f'LABELS_PUSH_429 retry_after={ra}s')
            return f'429:{ra}'
        log('LABELS_PUSH_ERR ' + repr(e)[:300])
        return 'err'

log('labels_push start')
# trigger: first push of the night has landed (v465 fork or track-B backup), + 10 min spacing
trig = wait_flag(['work/v465_pushed.txt', 'work/nb_sub_pushed.txt'], datetime.datetime(2026, 10, 25, 12, 0, 0))
if trig is None:
    log('no push flag by deadline; giving up'); raise SystemExit(3)
log(f'trigger={trig}; waiting 600s spacing before labels push')
time.sleep(600)
for round_no in range(1, 7):
    r = push_once()
    if r == 'ok':
        break
    if r.startswith('429'):
        wait = min(int(r.split(':')[1]) + 120, 24 * 3600)
        log(f'waiting {wait}s before labels round {round_no+1}')
        time.sleep(wait)
    else:
        time.sleep(1800)
else:
    log('labels push failed all rounds'); raise SystemExit(4)

# wait for kernel completion: CLI 'kaggle kernels status' works for own kernels
# (python-API GetKernelSessionStatus 403s); fall back to output polling.
import subprocess, glob, re
deadline = datetime.datetime.now() + datetime.timedelta(hours=4)
failures = 0
while datetime.datetime.now() < deadline:
    r = subprocess.run(['/usr/bin/timeout', '180', 'kaggle', 'kernels', 'status', 'lilac1996/casmi26-labels'], capture_output=True, text=True)
    m = re.findall(r'has status "([A-Za-z_.]*)"', r.stdout + r.stderr)
    st = m[-1].rsplit('.', 1)[-1].upper() if m else None
    if st:
        failures = 0
        log('kernel status ' + st)
        if st == 'COMPLETE':
            break
        if st in ('ERROR', 'CANCEL_REQUESTED', 'CANCEL_ACKNOWLEDGED'):
            log('labels kernel failed'); raise SystemExit(5)
    else:
        failures += 1
        log('status read failed (' + str(failures) + ') rc=' + str(r.returncode) + ' ' + (r.stdout + r.stderr)[-150:].replace('\n', ' | '))
        if failures % 10 == 0:
            rr = subprocess.run(['/usr/bin/timeout', '900', 'kaggle', 'kernels', 'output', 'lilac1996/casmi26-labels', '-p', 'work/labels_out',
                                 '--file-pattern', 'labels'], capture_output=True, text=True)
            if glob.glob('work/labels_out/**/labels*.csv', recursive=True):
                log('output present despite unknown status; treating as complete')
                break
    time.sleep(180)
man = 'work/labels_out/lilac1996/casmi26-labels/labels_manifest.json'
info = ''
if os.path.exists(man):
    info = open(man).read()
    log('MANIFEST ' + info)
open('work/labels_ready.txt', 'w').write('READY ' + datetime.datetime.now().isoformat() + '\n' + info + '\n')
log('labels_push done')
