"""LCMID ensemble: majority vote over __NCSV__ pipeline submissions (embedded labels)."""
import os
import pandas as pd

LABELS = __LABELS__
df = pd.DataFrame({'ID': list(range(680)), 'TARGET': LABELS})
df.to_csv('submission.csv', index=False)
print('ENSEMBLE_WRITTEN', len(df), df['TARGET'].value_counts().to_dict(), flush=True)

def submit_submission():
    SLUG = 'low-cost-motor-imagery-decoding-for-rehab'
    d = pd.read_csv('submission.csv')
    assert list(d.columns) == ['ID', 'TARGET'] and len(d) == 680
    assert (d['ID'].values == list(range(680))).all()
    assert set(d['TARGET'].unique()) <= {'rest', 'move'}
    print('schema OK:', d['TARGET'].value_counts().to_dict(), flush=True)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    res = api.competition_submit('submission.csv',
        '__NAME__: majority vote of __SRC__', SLUG, quiet=True)
    ref = getattr(res, 'ref', None)
    print('SUBMIT_RECEIPT ref=%s' % ref, flush=True)
    if not (isinstance(ref, int) and ref > 0):
        raise SystemExit('SUBMIT_FAILED: %r' % (res,))

if not os.environ.get('LCMID_LOCAL'):
    submit_submission()
