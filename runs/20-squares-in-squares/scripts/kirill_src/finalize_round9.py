"""Admit compact new witnesses only when better than frozen CUDA Packing Search V12."""
import csv,json,hashlib,math
from pathlib import Path
import numpy as np
from packing_geometry import load_bank,side,repair,save_bank,total_score
SCALE=100_000_000_000
if __name__=='__main__':
 baseline='inputs/round9/lightning/submission.csv';b=load_bank(baseline);chosen={n:p.copy()for n,p in b.items()};encoded={};changes=[]
 banks=[np.load(p)for p in ['artifacts/round9_axis_packings.npz','artifacts/round9_refined_packings.npz','artifacts/round9_portable_packings.npz','artifacts/round9_reclosure_packings.npz']if Path(p).exists()]
 for n in b:
  q=b[n]
  for z in banks:
   if str(n)in z and side(z[str(n)])<side(q)-1e-8:q=z[str(n)].copy()
  if (side(b[n])-side(q))/math.sqrt(n)<1e-7:continue
  q=repair(np.rint(q*SCALE)/SCALE,clearance=2e-9)
  if q is None:continue
  ints=np.rint(q*SCALE).astype(np.int64);q=ints.astype(float)/SCALE
  if (side(b[n])-side(q))/math.sqrt(n)<1e-7:continue
  chosen[n]=q;encoded[n]=ints;changes.append(dict(n=n,before=side(b[n]),after=side(q),gain=(side(b[n])-side(q))/math.sqrt(n),source='Seam LP + alternating contact refinement'))
 raw={}
 with open(baseline,newline='')as f:
  for r in csv.DictReader(f):raw.setdefault(int(r['n']),[]).append(r)
 out=Path('artifacts/round9_final_submission.csv');count=0
 with out.open('w',newline='')as f:
  w=csv.writer(f);w.writerow(['id','n','square_id','x','y','angle'])
  for n in sorted(raw):
   if n in encoded:
    for sid,(x,y,a)in enumerate(chosen[n]):w.writerow([count,n,sid,format(x,'.17g'),format(y,'.17g'),format(a,'.17g')]);count+=1
   else:
    for r in raw[n]:w.writerow([count,n,r['square_id'],r['x'],r['y'],r['angle']]);count+=1
 np.savez_compressed('artifacts/round9_patch_integers.npz',**{str(n):p for n,p in encoded.items()});save_bank('artifacts/round9_final_packings.npz',chosen)
 s=dict(baseline=total_score(b),candidate=total_score(chosen),gain=total_score(b)-total_score(chosen),changes=changes,new_changes=changes,patch_scale=SCALE,patch_ns=sorted(encoded),rows=count,sha256=hashlib.sha256(out.read_bytes()).hexdigest())
 Path('artifacts/round9_final_summary.json').write_text(json.dumps(s,indent=2));print({k:v for k,v in s.items()if k not in ['changes','new_changes','patch_ns']});print('Changed',len(changes))
