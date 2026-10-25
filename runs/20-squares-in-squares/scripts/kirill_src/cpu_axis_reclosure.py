"""CPU fixed-angle LP with randomized valid separation-axis choices.

Only square centres move. All admitted layouts retain exact unit-square sizes.
The radius covers every pair that could interact under bounded centre moves.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(key,'1')
import time,math
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from packing_geometry import normalize,side,repair

def reclose(p,seed=0,trust=.22,tie=.015,rounds=6,seconds=12):
    p=normalize(p);rng=np.random.default_rng(seed);start=time.perf_counter();history=[];n=len(p)
    for it in range(rounds):
        if time.perf_counter()-start>seconds:break
        L=side(p);a=p[:,2];c=np.cos(a);s=np.sin(a);h=(abs(c)+abs(s))/2
        radius=math.sqrt(2)+2*math.sqrt(2)*trust+1e-4
        pairs=cKDTree(p[:,:2]).query_pairs(radius,output_type='ndarray');i,j=pairs.T;d=p[j,:2]-p[i,:2]
        ux=np.column_stack((c[i],-s[i],c[j],-s[j]));uy=np.column_stack((s[i],c[i],s[j],c[j]));pr=d[:,0,None]*ux+d[:,1,None]*uy
        support=(1+abs(np.cos(a[i]-a[j]))+abs(np.sin(a[i]-a[j])))/2
        rank=abs(pr);valid=(rank>=support[:,None]-1e-9)&(rank>=rank.max(1)[:,None]-tie)
        random=rng.random(rank.shape);random[~valid]=-1;chosen=random.argmax(1);bad=~valid.any(1);chosen[bad]=rank[bad].argmax(1)
        rr=np.arange(len(i));sgn=np.sign(pr[rr,chosen]);vx=sgn*ux[rr,chosen];vy=sgn*uy[rr,chosen];dist=abs(pr[rr,chosen])
        cols=[np.column_stack((2*i,2*i+1,2*j,2*j+1))];vals=[np.column_stack((vx,vy,-vx,-vy))];rhs=[dist-support-3e-10]
        rows=[np.repeat(np.arange(len(i)),4)];offset=len(i);k=np.arange(n)
        for dim in [0,1]:
            for upper in [False,True]:
                cc=np.column_stack((2*k+dim,np.full(n,2*n)))
                vv=np.tile([1. if upper else -1.,-1. if upper else 0.],(n,1))
                bb=L-p[:,dim]-h if upper else p[:,dim]-h
                cols.append(cc);vals.append(vv);rhs.append(bb);rows.append(np.repeat(np.arange(offset,offset+n),2));offset+=n
        A=coo_matrix((np.concatenate([v.ravel()for v in vals]),(np.concatenate(rows),np.concatenate([v.ravel()for v in cols]))),shape=(offset,2*n+1)).tocsr()
        obj=np.zeros(2*n+1);obj[-1]=1
        fit=linprog(obj,A_ub=A,b_ub=np.concatenate(rhs),bounds=[(-trust,trust)]*(2*n)+[(-trust,trust)],method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'time_limit':max(.1,min(4.,seconds-(time.perf_counter()-start)))})
        if not fit.success:continue
        q=p.copy();q[:,:2]+=fit.x[:2*n].reshape(n,2);q=repair(q)
        if q is not None and side(q)<L-1e-8:
            p=q;history.append(dict(iteration=it,before=L,after=side(q),seed=seed,trust=trust,tie=tie))
    return p,history,time.perf_counter()-start
