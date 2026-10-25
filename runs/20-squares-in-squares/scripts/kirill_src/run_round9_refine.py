"""Refine the strongest seam/contact graph improvements over CUDA V12."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import json,time,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from packing_geometry import load_bank,side,save_bank,total_score
from cpu_axis_reclosure import reclose
from cpu_contact_polish import polish

def worker(job):
 n,p=job;before=side(p);h=[];start=time.perf_counter()
 for cycle in range(3):
  q,t,_=reclose(p,seed=117173+n+cycle*547,trust=[.55,.25,.4][cycle],tie=[.3,.06,.15][cycle],rounds=8,seconds=3)
  if side(q)<side(p):p=q
  q,u,_=polish(p,rounds=70,seed=16733+n+cycle,max_seconds=5,trust_start=[.04,.015,.025][cycle])
  if side(q)<side(p):p=q
  h.append(dict(cycle=cycle,after=side(p),axis=t,contact=u))
 return n,p,dict(n=n,before=before,after=side(p),gain=(before-side(p))/math.sqrt(n),seconds=time.perf_counter()-start,cycles=h)
if __name__=='__main__':
 z=np.load('artifacts/round9_axis_packings.npz');b={int(k):z[k]for k in z.files};h=json.loads(Path('artifacts/round9_axis_history.json').read_text());targets=list(dict.fromkeys([867,859,801,755,984,460,799,931,745,927,649,995]+[r['n']for r in sorted(h,key=lambda r:-r['gain'])[:4]]));print('Targets',targets,flush=True);history=[];start=time.perf_counter()
 with ProcessPoolExecutor(max_workers=2)as pool:
  for n,p,r in pool.map(worker,[(n,b[n])for n in targets]):
   history.append(r)
   if side(p)<side(b[n])-1e-8:b[n]=p
   save_bank('artifacts/round9_refined_packings.npz',b);Path('artifacts/round9_refined_history.json').write_text(json.dumps(history,indent=2));print(json.dumps({k:v for k,v in r.items()if k!='cycles'}),flush=True)
 print(json.dumps(dict(score=total_score(b),seconds=time.perf_counter()-start)),flush=True)
