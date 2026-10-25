"""Build a self-contained ensemble notebook dir: majority vote over given label CSVs.
Usage: python3 code/build_ens_nb.py <out_dir> <name> <id_suffix> <label_csv> [<label_csv> ...]
Priority order = argument order (used only for 2-way ties).
"""
import json, os, shutil, sys
import pandas as pd

out_dir, name, id_suffix = sys.argv[1], sys.argv[2], sys.argv[3]
csvs = sys.argv[4:]
assert len(csvs) >= 2, 'need >=2 csvs'
labels = []
for c in csvs:
    df = pd.read_csv(c)
    assert list(df.columns) == ['ID', 'TARGET'] and len(df) == 680
    assert (df['ID'].values == list(range(680))).all()
    labels.append(df['TARGET'].tolist())
votes = []
for i in range(680):
    cnt = {'rest': 0, 'move': 0}
    for lab in labels:
        cnt[lab[i]] += 1
    votes.append(('move' if cnt['move'] != cnt['rest'] and cnt['move'] > cnt['rest']
                  else ('rest' if cnt['rest'] != cnt['move'] else labels[0][i])))
n_move = votes.count('move')
print('ensemble: %d csvs, move=%d rest=%d' % (len(csvs), n_move, 680 - n_move))

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)
srcnames = ' + '.join(os.path.basename(os.path.dirname(c)) for c in csvs)
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ens_nb_template.py')).read()
script = (tpl.replace('__NCSV__', str(len(csvs)))
           .replace('__LABELS__', repr(votes))
           .replace('__NAME__', name)
           .replace('__SRC__', srcnames))
fn = os.path.join(out_dir, name + '.py')
open(fn, 'w').write(script)
meta = {
    "id": "lilac1996/lcmid-single-subject-" + id_suffix,
    "title": "LCMID " + name,
    "code_file": name + ".py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "false",
    "enable_tpu": "false",
    "enable_internet": "true",
    "competition_sources": ["low-cost-motor-imagery-decoding-for-rehab"],
    "dataset_sources": [],
    "kernel_sources": [],
    "model_sources": []
}
json.dump(meta, open(os.path.join(out_dir, 'kernel-metadata.json'), 'w'), indent=2)
print('built', out_dir)
