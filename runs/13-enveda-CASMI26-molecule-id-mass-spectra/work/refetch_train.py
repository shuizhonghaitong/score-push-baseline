import json, os, socket, subprocess, sys, time
RUN = '/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra'
os.chdir(RUN)
PROXY = 'http://172.28.4.16:18999'
OUT = 'data/enveda-casmi26-molecule-id-mass-spectra/train.parquet'
WANT = 3033286496
LOG = 'work/refetch_train.log'
def log(m):
    print(time.strftime('%F %T'), m, file=open(LOG, 'a'), flush=True)

log('refetch_train start (old signed URL expired 08:32 CST)')
url = None
for i in range(288):  # up to 24h, probe every 10min
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import ApiDownloadDataFileRequest
        socket.setdefaulttimeout(60)
        with kaggle.api.build_kaggle_client() as k:
            req = ApiDownloadDataFileRequest()
            req.competition_name = 'enveda-CASMI26-molecule-id-mass-spectra'
            req.file_name = 'train.parquet'
            resp = k.competitions.competition_api_client.download_data_file(req)
        url = resp.request.url
        json.dump({'train.parquet': url}, open('/tmp/casmi_urls.json', 'w'))
        log(f'URL acquired (probe {i+1}), len={len(url)}')
        break
    except Exception as e:
        msg = repr(e)
        tag = '429' if ('429' in msg or 'Too Many' in msg) else msg[:120]
        log(f'probe {i+1} failed: {tag}')
    time.sleep(600)
if not url:
    log('GAVE UP: no URL in 24h'); sys.exit(3)

for attempt in range(100000):
    have = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    if have >= WANT:
        log(f'DONE size={have}'); sys.exit(0)
    cmd = ['curl', '-sS', '--max-time', '900', '-C', str(have), '-o', OUT,
           '-w', '%{http_code}', '-x', PROXY, url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=960)
        code = (r.stdout or '').strip()[-3:]
        err = (r.stderr or '')[-120:]
    except subprocess.TimeoutExpired:
        code, err = 'TMO', 'python timeout'
    have2 = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    if attempt % 20 == 0 or code not in ('200', '206'):
        log(f'att {attempt+1} code={code} have {have}->{have2} err={err}')
    if have2 >= WANT:
        log(f'DONE size={have2}'); sys.exit(0)
    if code == '400' and have2 == have:
        # URL may have expired again mid-way; re-probe
        log('400 with no progress -> re-acquiring URL')
        try:
            import kaggle as k2
            from kaggle.api.kaggle_api_extended import ApiDownloadDataFileRequest as R2
            with k2.api.build_kaggle_client() as kk:
                rq = R2(); rq.competition_name = 'enveda-CASMI26-molecule-id-mass-spectra'; rq.file_name = 'train.parquet'
                url = kk.competitions.competition_api_client.download_data_file(rq).request.url
            json.dump({'train.parquet': url}, open('/tmp/casmi_urls.json', 'w'))
            log('URL re-acquired')
        except Exception as e:
            log('re-acquire failed: ' + repr(e)[:120])
        time.sleep(120)
        continue
    time.sleep(20)
