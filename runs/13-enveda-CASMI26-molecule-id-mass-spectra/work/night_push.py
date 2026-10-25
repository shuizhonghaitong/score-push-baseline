import kaggle, time, sys, json, datetime, os
LOG = 'work/night_push.log'
def log(msg):
    print(datetime.datetime.now().strftime('%F %T'), msg, file=open(LOG, 'a'), flush=True)

def push_once():
    import kaggle
    try:
        kaggle.api.kernels_push('work/nb_v465')
        log('PUSH_OK')
        open('work/v465_pushed.txt', 'w').write('PUSHED ' + datetime.datetime.now().isoformat() + '\n')
        return 'ok'
    except Exception as e:
        resp = getattr(e, 'response', None)
        if resp is not None and resp.status_code == 429:
            ra = int(resp.headers.get('retry-after', '3600'))
            log(f'PUSH_429 retry_after={ra}s (~{ra//3600}h{ra%3600//60}m)')
            return f'429:{ra}'
        log('PUSH_ERR ' + repr(e)[:200])
        return 'err'

target = datetime.datetime(2026, 10, 23, 2, 50, 0)
now = datetime.datetime.now()
delta = (target - now).total_seconds()
log(f'night_push start; sleeping {delta:.0f}s until {target}')
if delta > 0:
    time.sleep(delta)
for round_no in range(1, 8):
    r = push_once()
    if r == 'ok':
        break
    if r.startswith('429'):
        wait = min(int(r.split(':')[1]) + 120, 24 * 3600)
        log(f'waiting {wait}s before round {round_no+1}')
        time.sleep(wait)
    else:
        time.sleep(1800)
log('night_push done')
