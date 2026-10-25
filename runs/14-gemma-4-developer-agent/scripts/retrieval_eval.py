#!/usr/bin/env python3
"""Offline evaluation of file-localization strategies on the public task set.

For each task with a local snapshot: derive gold source files from the reference
patch, index non-test .py files, rank against the issue text with (a) TF-IDF,
(b) TF-IDF + mention boost, (c) grep-style identifier count. Report Hit@k and
gold rank per task/repo. CPU-only, reads .tgz members directly.
"""
import json, math, re, tarfile, time
from collections import Counter
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
TASKS = RUN / 'data' / 'tasks.jsonl'
SNAP = RUN / 'data' / 'snapshots'
OUT = RUN / 'results' / 'retrieval_eval'

_ID = re.compile(r'[A-Za-z_][A-Za-z0-9_]+')
_PART = re.compile(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+')
_TEST = re.compile(r'(^|/)(tests?/|testing/)|(^|/)test_[^/]*\.py$|_test\.py$')
_DIFF = re.compile(r'^(?:diff --git a/(\S+) b/.*|--- a/(\S+))', re.M)

def code_tokens(text):
    out = []
    for tok in _ID.findall(text):
        out.append(tok.lower())
        parts = [p.lower() for p in _PART.findall(tok) if len(p) > 1]
        if len(parts) > 1:
            out.extend(parts)
    return out

def gold_files(patch):
    gs = set()
    for m in _DIFF.finditer(patch):
        f = m.group(1) or m.group(2)
        if not f:
            continue
        f = f.strip()
        if _TEST.search(f) or f.startswith('docs/') or not f.endswith('.py'):
            continue
        gs.add(f.lstrip('./'))
    return gs

def read_py(entry):
    files = {}
    with tarfile.open(entry) as t:
        for mem in t:
            if not mem.isfile() or not mem.name.endswith('.py') or mem.size > 400_000:
                continue
            name = mem.name.lstrip('./')
            if _TEST.search(name):
                continue
            files[name] = t.extractfile(mem).read().decode('utf-8', 'ignore')
    return files

def build(files):
    paths = sorted(files)
    n = len(paths)
    df = Counter()
    docs_tok = []
    for p in paths:
        toks = code_tokens(p.replace('/', ' ') * 3) + code_tokens(files[p][:200_000])
        docs_tok.append(toks)
        df.update(set(toks))
    idf = {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}
    docs = []
    for toks in docs_tok:
        tf = Counter(toks)
        vec = {w: (1.0 + math.log(c)) * idf[w] for w, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        docs.append({w: v / norm for w, v in vec.items()})
    defs = [set(re.findall(r'^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)', files[p], re.M)) for p in paths]
    return paths, docs, defs

def score_query(files, paths, docs, query):
    q = Counter(code_tokens(query))
    scores = [0.0] * len(paths)
    for qi, (w, c) in enumerate(q.items()):
        pass
    for i, d in enumerate(docs):
        s = sum(qw * d[w] for w, qw in q.items() if w in d)
        scores[i] = s
    return scores

def boost_scores(paths, defs, scores, query):
    ql = query.lower()
    q_ids = {t for t in _ID.findall(query) if len(t) >= 4 and ('_' in t or (len(t) > 1 and t[1].isupper()))}
    out = list(scores)
    for i, p in enumerate(paths):
        stem = p.rsplit('/', 1)[-1][:-3]
        if p.lower() in ql or (len(stem) >= 4 and re.search(r'\b' + re.escape(stem.lower()) + r'\b', ql)):
            out[i] += 1.0
        out[i] += min(1.0, 0.35 * len(q_ids & defs[i]))
    return out

def grep_scores(files, paths, query):
    pats = []
    for t in _ID.findall(query):
        if len(t) >= 4 and ('_' in t or (len(t) > 1 and t[1].isupper()) or t.lower() not in {'def','class','return','import','from','with','async','await','self','lambda','yield'}):
            pats.append(re.compile(r'\b' + re.escape(t) + r'\b').search)
    scores = []
    for p in paths:
        content = files[p][:200_000]
        scores.append(sum(1 for s in pats if s(content)))
    return scores

_STOP = {'def','class','return','import','from','with','async','await','self','lambda','yield','the','and','for','not','you','that','this','are','was','have','has','will','can','should','when','where','which','would','could','been','being','use','used','using','get','set','new','type','value','values','name','names','function','method','parameter','parameters','argument','arguments','string','list','dict','dict','int','float','bool','None','true','false','error','errors','issue','issues','bug','fix','add','adds','added','support','supports','supporting','return','returns','raise','raises','raised','call','calls','called','create','creates','created','file','files','test','tests','case','cases','function','module','modules','class','classes','object','objects','value','values','key','keys','item','items','data','api','url','http','request','requests','response','responses','server','client','code','test','tests'}
def words_scores(files, paths, query):
    toks = [t for t in (w.lower() for w in _ID.findall(query)) if len(t) >= 3 and t not in _STOP]
    pats = [re.compile(r'\b' + re.escape(t) + r'\b').search for t in dict.fromkeys(toks)]
    return [sum(1 for s in pats if s(files[p][:200_000])) for p in paths]

def norm(s):
    m = max(s) if s else 0.0
    return [x / m if m else 0.0 for x in s]

def gold_rank(paths, gold):
    best = None
    for k, p in enumerate(paths, 1):
        if any(p == g or p.endswith('/' + g) or g.endswith('/' + p) for g in gold):
            best = k
            break
    return best

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for line in open(TASKS):
        t = json.loads(line)
        iid = t['instance_id']
        snap = SNAP / (iid + '.tgz')
        if not snap.exists() or not t.get('patch'):
            continue
        gold = gold_files(t['patch'])
        if not gold:
            continue
        try:
            files = read_py(snap)
            if len(files) < 3:
                rows.append(dict(id=iid, repo=t['repo'], n=len(files), skip='too-few-files'))
                continue
            paths, docs, defs = build(files)
            query = t['problem_statement'] + ('\n' + t['hints_text'] if t.get('hints_text') else '')
            s_tf = score_query(files, paths, docs, query)
            s_b = boost_scores(paths, defs, s_tf, query)
            s_g = grep_scores(files, paths, query)
            s_w = words_scores(files, paths, query)
            nt, ng, nw = norm(s_tf), norm(s_g), norm(s_w)
            s_f1 = [nt[i] + ng[i] for i in range(len(paths))]
            s_f2 = [nt[i] + 1.5*ng[i] + nw[i] for i in range(len(paths))]
            s_f3 = [s_b[i] - s_tf[i] + 1.0 for i in range(len(paths))]  # mention+defs without tfidf (boost-tf)
            order = lambda s: [paths[i] for i in sorted(range(len(paths)), key=lambda i: -s[i])]
            row = dict(id=iid, repo=t['repo'], n=len(files), nq=len(gold),
                       r_tf=gold_rank(order(s_tf), gold),
                       r_b=gold_rank(order(s_b), gold),
                       r_g=gold_rank(order(s_g), gold),
                       r_w=gold_rank(order(s_w), gold),
                       r_f1=gold_rank(order(s_f1), gold),
                       r_f2=gold_rank(order(s_f2), gold),
                       r_f3=gold_rank(order(s_f3), gold))
            rows.append(row)
            print(f"{time.time()-t0:6.0f}s {iid} n={len(files)} tf={row['r_tf']} boost={row['r_b']} grep={row['r_g']} gold={sorted(gold)[:2]}", flush=True)
        except Exception as e:
            rows.append(dict(id=iid, repo=t['repo'], skip=f'err:{type(e).__name__}:{str(e)[:80]}'))
            print(f"{time.time()-t0:6.0f}s {iid} ERROR {type(e).__name__} {str(e)[:100]}", flush=True)
    (OUT / 'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    def agg(key):
        ks = [1, 3, 5, 10]
        vals = [r[key] for r in rows if r.get(key) is not None]
        hit = {k: sum(1 for v in vals if v <= k) / len(vals) for k in ks}
        med = sorted(vals)[len(vals)//2] if vals else None
        return len(vals), med, hit
    for key, name in [('r_tf','tfidf'), ('r_b','boost'), ('r_g','grep'), ('r_w','words'), ('r_f1','tf+grep'), ('r_f2','tf+1.5grep+words'), ('r_f3','boost-notf')]:
        n, med, hit = agg(key)
        print(f"\n{name}: n={n} median_rank={med} " + ' '.join(f"Hit@{k}={v:.2f}" for k, v in hit.items()))
    repos = {}
    for r in rows:
        if r.get('r_b') is not None:
            repos.setdefault(r['repo'], []).append(r['r_b'])
    for repo, vals in sorted(repos.items()):
        h1 = sum(1 for v in vals if v <= 1) / len(vals)
        h5 = sum(1 for v in vals if v <= 5) / len(vals)
        print(f"  {repo}: n={len(vals)} Hit@1={h1:.2f} Hit@5={h5:.2f} median={sorted(vals)[len(vals)//2]}")

if __name__ == '__main__':
    main()
