import json, os, subprocess, sys, time
RUN = '/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra'
ICE = f'{RUN}/data/local/iceberg'
GL  = f'{RUN}/data/local/glacier'
SITE = f'{RUN}/data/local/site'
OUTD = f'{RUN}/work/canary'
os.makedirs(OUTD, exist_ok=True)
PY = sys.executable

import torch
assert torch.cuda.is_available(), 'CUDA unavailable in local training job'
print('torch', torch.__version__, 'cuda device', torch.cuda.get_device_name(0), flush=True)

inp = f'{OUTD}/canary_in.json'
jobs = json.load(open(inp))
print('jobs:', [(j['mid'], len(j['cands'])) for j in jobs], flush=True)

def run_sim(name, runner, pkg, out, extra):
    t0 = time.time()
    r = subprocess.run([PY, runner, pkg, inp, out,
                        '--device', 'cuda', '--site', SITE, '--budget', '600', '--quiet', '--any-rdkit'] + extra,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(f'{name} rc={r.returncode} t={time.time()-t0:.1f}s', flush=True)
    print(r.stdout[-2500:], flush=True)
    res = json.load(open(out))
    meta = json.load(open(out + '.meta.json'))
    nonnull = sum(1 for v in res.values() if isinstance(v, dict) for s in v.values() if s is not None)
    print(f'{name} status={meta.get("status")} device={meta.get("device")} torch={meta.get("torch")} rdkit={meta.get("rdkit")} non_null={nonnull}', flush=True)
    print(f'{name} sample:', {k: v for k, v in list(res.items())[:1]}, flush=True)
    assert meta.get('status') == 'ok' or nonnull > 0, f'{name} failed: {meta}'

run_sim('ICE', f'{ICE}/ice_runner.py', ICE, f'{OUTD}/ice_out.json', [])
run_sim('GL', f'{GL}/gl_runner.py', GL, f'{OUTD}/gl_out.json', ['--wheels', f'{ICE}/wheels'])
print('CANARY_OK', flush=True)
