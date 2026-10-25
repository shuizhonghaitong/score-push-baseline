import kaggle, time, sys
for i in range(4):
    t0 = time.time()
    try:
        kaggle.api.kernels_push('work/nb_v465')
        print(f'attempt {i}: PUSH_OK in {time.time()-t0:.0f}s', flush=True)
        sys.exit(0)
    except Exception as e:
        resp = getattr(e, 'response', None)
        if resp is not None:
            h = {k: v for k, v in resp.headers.items() if k.lower() in ('retry-after','x-ratelimit-remaining','x-ratelimit-limit','server','cf-ray','x-request-id')}
            print(f'attempt {i}: {resp.status_code} hdrs={h} {repr(e)[:80]}', flush=True)
        else:
            print(f'attempt {i}: connerr {repr(e)[:110]}', flush=True)
    time.sleep(45)
