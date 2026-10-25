import glob, os, subprocess, sys, time, hashlib
RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RUN)
WHEEL = 'data/local/wheels311/rdkit-2025.3.6-cp311-cp311-manylinux_2_28_x86_64.whl'
SHA = 'afc931d1be1a4ad2fc4f2f2732c10983b1a57fbde9c22870721beafe42be3ab0'
SITE = 'data/local/site'
LOG = 'work/finish_site.log'
def log(m):
    print(time.strftime('%F %T'), m, file=open(LOG, 'a'), flush=True)

def sha_ok():
    try:
        return hashlib.sha256(open(WHEEL, 'rb').read()).hexdigest() == SHA
    except Exception:
        return False

# 1) wait for the wheel
while not sha_ok():
    log('waiting for rdkit wheel...')
    time.sleep(300)
log('wheel sha ok')

# 2) install rdkit into the existing site
r = subprocess.run(['/mnt/disk3/liuyutong_home/miniconda3/envs/qwen/bin/python', '-m', 'pip', 'install',
                    '--no-index', '--no-deps', '--no-compile', '--disable-pip-version-check', '--target', SITE, WHEEL],
                   capture_output=True, text=True)
log('pip rc=' + str(r.returncode) + ' ' + r.stdout[-200:])
if r.returncode != 0:
    log('FAILED pip install'); sys.exit(1)

# 3) marker matching what ice_runner computes for iceberg/wheels
src = sorted(glob.glob('data/local/iceberg/wheels/*.whl.ice') + glob.glob('data/local/iceberg/wheels/*.whl'))
open(os.path.join(SITE, '.ice_site_ok'), 'w').write('\n'.join(os.path.basename(w) for w in src))
log('marker written for ' + str(len(src)) + ' wheels')

# 4) verify rdkit version from the site
r = subprocess.run(['/mnt/disk3/liuyutong_home/miniconda3/envs/qwen/bin/python', '-c',
                    f'import sys; sys.path.insert(0, "{os.path.abspath(SITE)}"); import rdkit; print("RDKIT", rdkit.__version__)'],
                   capture_output=True, text=True)
log('rdkit check: ' + r.stdout.strip() + ' ' + r.stderr[-200:])
open('work/site_ready.txt', 'w').write(time.strftime('%F %T') + ' ' + r.stdout.strip() + '\n')

# 5) submit canary job
r = subprocess.run(['/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python',
                    '/mnt/disk3/liuyutong_home/workspace/skill/score-push/tools/local_training.py',
                    'submit', '--run-dir', RUN, '--memory-gib', '16', 'work/canary_sim.py'],
                   capture_output=True, text=True)
log('canary submit rc=' + str(r.returncode) + ' ' + r.stdout.strip() + ' ' + r.stderr[-200:])
sys.exit(0 if r.returncode == 0 else 1)
