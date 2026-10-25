"""Build the 680-row submission from an ensemble of ML configs."""
import sys, os, argparse, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_subject, TEST_SUBJECTS
import features as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import eval_cv as E

def predict_subject(sub, configs, threshold=0.5):
    """configs: list of (feat, model_kind). Returns averaged prob for 40 test epochs."""
    d = load_subject(sub)
    runs = [k for k in ('tr01', 'tr02', 'tr03') if k in d]
    cues = np.concatenate([d[k][0] for k in runs])
    bases = np.concatenate([d[k][1] for k in runs])
    y = np.concatenate([d[k][2] for k in runs])
    tcues, tbases, _, _ = d['test']
    probs = []
    for fn, mk in configs:
        if fn in ('bp', 'td', 'bp_td'):
            Xtr = E.static_featurize(fn, cues, bases)
            Xva = E.static_featurize(fn, tcues, tbases)
        else:
            Xtr, state = E.featurize(fn, cues, bases, y, None)
            Xva = E.featurize(fn, tcues, tbases, None, state)
        clf = make_pipeline(StandardScaler(), E.make_model(mk))
        clf.fit(Xtr, y)
        probs.append(clf.predict_proba(Xva)[:, 1])
    return np.mean(probs, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', default='bp:lda_0.1,bp:lda_0.3,bp:lr_0.1,bp_csp:lda_0.3,bp_rts:lda_0.3')
    ap.add_argument('--out', default='submissions/submission_ml_v1.csv')
    ap.add_argument('--probs-out', default=None)
    args = ap.parse_args()
    configs = [tuple(c.split(':')) for c in args.configs.split(',')]
    rows = []
    probs_all = {}
    i = 0
    for sub in TEST_SUBJECTS:
        p = predict_subject(sub, configs)
        labels = ['move' if q >= 0.5 else 'rest' for q in p]
        probs_all[sub] = p.tolist()
        for lab in labels:
            rows.append((i, lab))
            i += 1
    assert i == 680, i
    import pandas as pd
    df = pd.DataFrame(rows, columns=['ID', 'TARGET'])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df['TARGET'].value_counts().to_dict())
    print('wrote', args.out)
    if args.probs_out:
        json.dump(probs_all, open(args.probs_out, 'w'))
        print('wrote', args.probs_out)

if __name__ == '__main__':
    main()
