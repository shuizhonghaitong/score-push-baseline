"""A bounded fixed-angle separation-graph scan on all submitted layouts."""
from pathlib import Path
import json,time,math
from packing_geometry import load_bank,side,save_bank,total_score
from cpu_axis_reclosure import reclose
if __name__=='__main__':
    b=load_bank('inputs/round9/lightning/submission.csv');h=[];start=time.perf_counter()
    targets=[n for n in b if side(b[n])<math.ceil(math.sqrt(n))-1e-7]
    for idx,n in enumerate(targets):
        before=side(b[n]);q,t,sec=reclose(b[n],seed=41621+n,trust=.45,tie=.2,rounds=6,seconds=1.5)
        if side(q)<before-1e-7:
            b[n]=q;record=dict(n=n,before=before,after=side(q),gain=(before-side(q))/math.sqrt(n),trace=t,seconds=sec);h.append(record);print(json.dumps(record),flush=True)
        if idx%30==0:
            print(json.dumps(dict(progress=idx,parents=len(targets),hits=len(h),seconds=time.perf_counter()-start)),flush=True);save_bank('artifacts/round9_axis_packings.npz',b);Path('artifacts/round9_axis_history.json').write_text(json.dumps(h,indent=2))
    save_bank('artifacts/round9_axis_packings.npz',b);Path('artifacts/round9_axis_history.json').write_text(json.dumps(h,indent=2));print(json.dumps(dict(score=total_score(b),seconds=time.perf_counter()-start)),flush=True)
