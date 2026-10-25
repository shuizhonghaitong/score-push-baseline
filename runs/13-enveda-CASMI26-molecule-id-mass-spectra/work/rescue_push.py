import kaggle, time, datetime, os, re
RUN = '/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra'
os.chdir(RUN)
def log(msg):
    print(datetime.datetime.now().strftime('%F %T'), msg, file=open('work/rescue.log', 'a'), flush=True)

def smoke_ok():
    try:
        txt = open('work/auto_v465.log').read()
        m = re.findall(r'READY_TO_SUBMIT VERSION=\S* SUB=(\S*) verify_rc=(\d+)', txt)
        if m and m[-1][1] == '0' and m[-1][0] and os.path.exists(m[-1][0]):
            return True
        if os.path.exists('work/sequencer.log') and 'ok=True' in open('work/sequencer.log').read():
            return True
    except Exception:
        pass
    return False

target = datetime.datetime(2026, 10, 24, 9, 0)
log(f'rescue start; checking at {target}')
while datetime.datetime.now() < target:
    time.sleep(300)
if smoke_ok():
    log('smoke already verified -> nothing to do')
else:
    log('smoke NOT verified -> rescue push of work/nb_v465')
    for i in range(6):
        try:
            kaggle.api.kernels_push('work/nb_v465')
            log('RESCUE_PUSH_OK')
            open('work/v465_pushed.txt', 'a').write('RESCUE ' + datetime.datetime.now().isoformat() + '\n')
            break
        except Exception as e:
            resp = getattr(e, 'response', None)
            if resp is not None and resp.status_code == 429:
                ra = int(resp.headers.get('retry-after', '1800'))
                log(f'RESCUE 429 retry_after={ra}')
                time.sleep(min(ra + 120, 4 * 3600))
            else:
                log('RESCUE_ERR ' + repr(e)[:200])
                time.sleep(900)
log('rescue done')
