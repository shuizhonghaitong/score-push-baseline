"""Submit the ML v1 pipeline's submission.csv to the competition via the Kaggle API (in-kernel)."""
import os, glob, json, warnings
warnings.filterwarnings('ignore')

SLUG = 'low-cost-motor-imagery-decoding-for-rehab'
MESSAGE = 'ML v1: per-subject ensemble bp+rts+csp features, LogReg/LDA'

# locate the pipeline output: kernel outputs are mounted under /kaggle/input/kernels/
found = None
cands = []
for root, dirs, files in os.walk('/kaggle/input'):
    dirs[:] = [d for d in dirs if d != 'competitions']
    if 'submission.csv' in files:
        cands.append(os.path.join(root, 'submission.csv'))
for p in cands:
    print('candidate:', p)
if cands:
    found = cands[0]
if found is None and os.path.exists('submission.csv'):
    found = 'submission.csv'
if found is None:
    raise SystemExit('SUBMISSION_FILE_NOT_FOUND. /kaggle/input: %r' % os.listdir('/kaggle/input'))

import pandas as pd
df = pd.read_csv(found)
assert list(df.columns) == ['ID', 'TARGET'] and len(df) == 680, (list(df.columns), len(df))
assert (df['ID'].values == list(range(680))).all()
assert set(df['TARGET'].unique()) <= {'rest', 'move'}
print('schema OK:', df['TARGET'].value_counts().to_dict())
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
res = api.competition_submit(found, MESSAGE, SLUG, quiet=True)
ref = getattr(res, 'ref', None)
print('SUBMIT_RECEIPT ref=%s' % ref)
if not (isinstance(ref, int) and ref > 0):
    raise SystemExit('SUBMIT_FAILED: %r' % (res,))
