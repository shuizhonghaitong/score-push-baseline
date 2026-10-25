import pandas as pd, glob, os
from collections import Counter

DATA = 'data'
tr = sorted(glob.glob(f'{DATA}/Training set/*.csv'))
te = sorted(glob.glob(f'{DATA}/Single subject test set/*.csv'))
print('train files:', len(tr), 'test files:', len(te))

# marker vocabulary per file type
tr_markers = Counter()
for f in tr:
    m = pd.read_csv(f, usecols=['Marker_val'])['Marker_val'].dropna()
    tr_markers.update(m.astype(str).tolist())
te_markers = Counter()
for f in te:
    m = pd.read_csv(f, usecols=['Marker_val'])['Marker_val'].dropna()
    te_markers.update(m.astype(str).tolist())
print('TRAIN markers:', dict(tr_markers))
print('TEST markers:', dict(te_markers))

def seq(fname, kind):
    df = pd.read_csv(fname, usecols=['time','Marker_val'])
    m = df['Marker_val'].dropna()
    idx = m.index.tolist()
    vals = m.astype(str).tolist()
    # sequence pattern of first 15 markers
    print(fname.split('/')[-1], 'n_markers=', len(m))
    print(' first markers:', vals[:15])
    print(' last markers:', vals[-5:])
    if kind=='test':
        # count cue_start and gaps between cue_start and following cue_stop
        starts = [i for i,v in enumerate(vals) if v=='cue_start']
        stops = [i for i,v in enumerate(vals) if v=='cue_stop']
        print(' n_cue_start=', len(starts), ' n_cue_stop=', len(stops))
        t = df['time'].values
        if starts and stops:
            durs = []
            for s in starts[:3]:
                i = stops.index([i for i in stops if i>starts[s]][0]) if any(i>starts[s] for i in stops) else None
            print(' time range:', t[0], t[-1], ' dur_s=', (t[-1]-t[0]))
        # durations cue_start->cue_stop in samples
        durs = []
        si = 0
        for s in starts:
            cands = [i for i in stops if i > s]
            if cands:
                durs.append(cands[0]-s)
        print(' cue durations (rows): min', min(durs), 'max', max(durs), 'mean', sum(durs)/len(durs))
        # sample rate
        dt = pd.Series(t).diff().median()
        print(' median dt_ms:', dt*1000)

seq(tr[0], 'train')
print()
seq(te[0], 'test')
print()
# check all test files have 40 cue_start
for f in te:
    m = pd.read_csv(f, usecols=['Marker_val'])['Marker_val'].dropna().astype(str)
    n = (m=='cue_start').sum()
    print(f.split('/')[-1], 'cue_start:', n)
