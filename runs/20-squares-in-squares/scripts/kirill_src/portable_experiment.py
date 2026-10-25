"""Recompute using an explicitly chosen tensor device; certify every accepted row.

The tensor proposal stage runs on cpu/cuda/mps. Exact LP polishing, decimal
serialization and the unchanged official metric run on CPU in every mode.
"""
import csv,json,math,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from packing_geometry import load_bank,side,repair
from portable_seam_search import seam_pressure,choose_device
from cpu_axis_reclosure import reclose
from cpu_contact_polish import polish

def run_experiment(input_csv,output_csv,backend='cpu',sizes=(801,460),batch=6,steps=240):
    choose_device(backend)  # Fail explicitly; never silently switch devices.
    sys.path.insert(0,str(Path(__file__).resolve().parent/'reference_sources'))
    import official_metric as metric
    bank=load_bank(input_csv);rows=defaultdict(list)
    with Path(input_csv).open(newline='')as f:
        for row in csv.DictReader(f):rows[int(row['n'])].append(row)
    history=[]
    for n in sizes:
        if n not in rows:raise ValueError('Choose a submitted non-grid puzzle size.')
        before=metric._evaluate_puzzle_worker((n,rows[n]));assert before[3]=='ok'
        proposals,report=seam_pressure(bank[n],backend=backend,seed=99173+n,batch=batch,steps=steps)
        q,_,_=reclose(bank[n],seed=397+n,rounds=6,seconds=3,trust=.3,tie=.1);proposals.append(q)
        best_rows=rows[n];best_score=before[1]
        for q in sorted(proposals,key=side)[:3]:
            if side(q)>side(bank[n])+.015:continue
            q,_,_=polish(q,rounds=60,max_seconds=6,trust_start=.015)
            q=repair(q,clearance=2e-9)
            if q is None:continue
            q=np.rint(q*1e11)/1e11;q=repair(q,clearance=2e-9)
            candidate=[dict(id=str(sid),n=str(n),square_id=str(sid),x=format(x,'.17g'),y=format(y,'.17g'),angle=format(a,'.17g'))for sid,(x,y,a)in enumerate(q)]
            result=metric._evaluate_puzzle_worker((n,candidate))
            if result[3]=='ok'and result[1]<best_score-1e-8:
                # Also reject any candidate that changes validity when parsed
                # with pandas' ordinary float64 parser and serialized again.
                import pandas as pd,io
                buf=io.StringIO();pd.read_csv(io.StringIO(pd.DataFrame(candidate).to_csv(index=False))).to_csv(buf,index=False,float_format='%.17g');buf.seek(0)
                rt=metric._evaluate_puzzle_worker((n,list(csv.DictReader(buf))))
                if rt[3]!='ok'or abs(rt[1]-result[1])>1e-10:continue
                best_rows=candidate;best_score=result[1]
        rows[n]=best_rows;report.update(n=n,official_before=before[1],official_after=best_score,accepted=best_score<before[1]-1e-8);history.append(report)
    fields=['id','n','square_id','x','y','angle'];index=0
    with Path(output_csv).open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for n in sorted(rows):
            for row in rows[n]:row=dict(row);row['id']=str(index);writer.writerow(row);index+=1
    Path('recompute_history.json').write_text(json.dumps(history,indent=2))
    return history

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--backend',choices=['cpu','cuda','mps'],default='cpu')
    parser.add_argument('--sizes',nargs='+',type=int,default=[801,460]);parser.add_argument('--batch',type=int,default=6);parser.add_argument('--steps',type=int,default=240)
    args=parser.parse_args()
    history=run_experiment(args.input,args.output,backend=args.backend,sizes=args.sizes,batch=args.batch,steps=args.steps)
    print(json.dumps([dict(n=r['n'],backend=r['backend'],accepted=r['accepted'],before=r['official_before'],after=r['official_after'])for r in history],indent=2))
