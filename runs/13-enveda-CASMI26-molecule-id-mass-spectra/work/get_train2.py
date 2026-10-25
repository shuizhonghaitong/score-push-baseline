"""Multi-stream downloader with 200-full-stream detection.
If the proxy ever ignores Range (HTTP 200), the stream contains the WHOLE file:
detect it (size reaches WANT) and use it as the final train.parquet.
"""
import json, os, subprocess, time, sys
import concurrent.futures as cf
PROXY = 'http://172.28.4.16:18999'
URL = json.load(open('/tmp/casmi_urls.json'))['train.parquet']
WANT = 3033286496
N = 4
DIR = 'data/enveda-casmi26-molecule-id-mass-spectra'
PARTS = DIR + '/train2.parquet.parts'
os.makedirs(PARTS, exist_ok=True)
def log(m): print(time.strftime('%F %T'), m, file=open('work/get_train2.log', 'a'), flush=True)
chunk = (WANT + N - 1) // N

stop = {'flag': False}
def fetch(i):
    a = i * chunk
    b = min(WANT - 1, a + chunk - 1)
    want = b - a + 1
    part = os.path.join(PARTS, f'{i:03d}.bin')
    for attempt in range(100000):
        if stop['flag']:
            return 'stop'
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have >= want:
            return 'ok'
        if have > want:
            log(f'part {i} oversized {have}>{want}; starting fresh')
            have = 0
            if os.path.exists(part):
                os.remove(part)
        cmd = ['curl', '-sS', '-f', '-r', f'{a + have}-{b}', '-x', PROXY, '-o', part,
               '-w', '%{http_code}', '--max-time', '600', URL]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=660)
            code = (r.stdout or '').strip()[-3:]
        except subprocess.TimeoutExpired:
            code = 'TMO'
        have2 = os.path.getsize(part) if os.path.exists(part) else 0
        if code == '200':
            # proxy ignored range: this file may be the ENTIRE dataset
            if have2 == WANT:
                log(f'part {i} got FULL 200 stream ({have2}); promoting to final')
                dst = DIR + '/train.parquet'
                os.replace(part, dst)
                stop['flag'] = True
                return 'full200'
            log(f'part {i} 200 but size {have2} (truncated mid-stream); continuing as fresh range file')
            os.remove(part) if have2 else None
        elif code == '206' and have2 > have:
            pass
        else:
            log(f'part {i} att {attempt+1} code={code} have {have}->{have2}')
            time.sleep(15 + (attempt % 10) * 5)
    return 'failed'

log(f'start N={N} chunk={chunk}')
with cf.ThreadPoolExecutor(N) as ex:
    futs = [ex.submit(fetch, i) for i in range(N)]
    results = [f.result() for f in futs]
log('results ' + str(results))
if 'full200' in results:
    log('FULL200 DONE')
    sys.exit(0)
if all(r == 'ok' for r in results):
    dst = DIR + '/train.parquet'
    if os.path.exists(dst):
        os.remove(dst)
    with open(dst, 'wb') as fo:
        for i in range(N):
            with open(os.path.join(PARTS, f'{i:03d}.bin'), 'rb') as fi:
                while True:
                    blk = fi.read(1 << 24)
                    if not blk:
                        break
                    fo.write(blk)
    log(f'assembled {dst} {os.path.getsize(dst)}')
else:
    log('INCOMPLETE ' + str(results))
    sys.exit(1)
