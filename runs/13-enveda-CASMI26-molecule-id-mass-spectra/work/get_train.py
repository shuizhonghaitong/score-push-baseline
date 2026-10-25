"""Single-stream resume downloader for the signed GCS train.parquet.
Tolerates: 502 waves, 2h blackholes (short curl timeout + retry), proxy dropping the
Range header (200 full-stream response -> we simply get the whole file, which is fine).
"""
import json, os, subprocess, time, sys
PROXY = 'http://172.28.4.16:18999'
URL = json.load(open('/tmp/casmi_urls.json'))['train.parquet']
OUT = 'data/enveda-casmi26-molecule-id-mass-spectra/train.parquet'
WANT = 3033286496
def log(m): print(time.strftime('%F %T'), m, file=open('work/get_train.log', 'a'), flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
log(f'start; url age ok?; want={WANT}')
for attempt in range(100000):
    have = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    if have >= WANT:
        log(f'DONE size={have}')
        sys.exit(0)
    if have > WANT:
        log(f'oversized file {have} > {WANT}; will re-request (200 path will overwrite)')
    cmd = ['curl', '-sS', '--max-time', '900',
           '-C', str(have), '-o', OUT, '-w', '%{http_code}', '-x', PROXY, URL]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=960)
        code = (r.stdout or '').strip()[-3:]
        err = (r.stderr or '')[-120:]
    except subprocess.TimeoutExpired:
        code, err = 'TMO', 'python timeout'
    have2 = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    log(f'att {attempt+1} code={code} have {have}->{have2} err={err}')
    if have2 >= WANT:
        log(f'DONE size={have2}')
        sys.exit(0)
    time.sleep(20)
