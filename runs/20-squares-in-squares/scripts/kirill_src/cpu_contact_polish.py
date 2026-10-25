# Adapted from lightningv08 CUDA Packing Search contact_polish.py (Apache 2.0).
# Changes: CPU-only imports, configurable regularization/trust, local experiment runner.
"""Trust-region sequential LP with free centers/angles and exact SAT admission.

Background: coupled sparse LP refinement in prvsiyan's A Square Deal notebook.
This vectorized constraint builder and experiment runner are implemented here.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,math,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from packing_geometry import side, normalize, repair

def polish(p,rounds=45,seed=0,max_seconds=25,regularization=0.,trust_start=.04):
    p=normalize(p);n=len(p);rng=np.random.default_rng(seed);at=time.perf_counter();step=trust_start;trace=[]
    for it in range(rounds):
        if time.perf_counter()-at>max_seconds:break
        L=side(p);theta=p[:,2];c=np.cos(theta);s=np.sin(theta);h=(abs(c)+abs(s))/2
        edges=cKDTree(p[:,:2]).query_pairs(1.75,output_type='ndarray');i,j=edges.T;d=p[j,:2]-p[i,:2]
        ux=np.column_stack((c[i],-s[i],c[j],-s[j]));uy=np.column_stack((s[i],c[i],s[j],c[j]));proj=d[:,0,None]*ux+d[:,1,None]*uy
        ranks=abs(proj);rank=ranks+rng.uniform(-2e-12,2e-12,ranks.shape);chosen=rank.argmax(1);rr=np.arange(len(i))
        sign=np.sign(proj[rr,chosen]);vx=sign*ux[rr,chosen];vy=sign*uy[rr,chosen];rotation=-d[:,0]*vy+d[:,1]*vx
        delta=theta[j]-theta[i];cd=np.cos(delta);sd=np.sin(delta);rad=(1+abs(cd)+abs(sd))/2;dr=(-sd*np.sign(cd)+cd*np.sign(sd))/2
        dist=abs(proj[rr,chosen]);own=chosen<2
        columns=np.column_stack((3*i,3*i+1,3*j,3*j+1,3*i+2,3*j+2))
        values=np.column_stack((vx,vy,-vx,-vy,-dr-np.where(own,rotation,0),dr-np.where(own,0,rotation)))
        rows=[];cols=[];vals=[];bounds=[];offset=0
        def add(cc,vv,bb):
            nonlocal offset
            bb=np.asarray(bb);count=len(bb)
            rows.append(np.repeat(np.arange(offset,offset+count),cc.shape[1]));cols.append(cc.ravel());vals.append(vv.ravel());bounds.append(bb);offset+=count
        add(columns,values,dist-rad-4e-11)
        # At relative right angles, both one-sided radius derivatives are needed.
        kink=np.flatnonzero(np.minimum(abs(cd),abs(sd))<1e-8)
        for slope in [-.5,.5]:
            vv=values[kink].copy();vv[:,4]=-slope-np.where(own[kink],rotation[kink],0);vv[:,5]=slope-np.where(own[kink],0,rotation[kink])
            add(columns[kink],vv,dist[kink]-rad[kink]-4e-11)
        k=np.arange(n);dh=(-s*np.sign(c)+c*np.sign(s))/2
        aligned=np.minimum(abs(c),abs(s))<1e-8
        for dim in [0,1]:
            for upper in [False,True]:
                ids=k;extra=k[aligned]
                for ids,deriv in [(ids,dh),(extra,np.full(len(extra),-.5)),(extra,np.full(len(extra),.5))]:
                    cc=np.column_stack((3*ids+dim,3*ids+2,np.full(len(ids),3*n)))
                    vv=np.column_stack((np.full(len(ids),1. if upper else -1.),deriv,np.full(len(ids),-1. if upper else 0.)))
                    bb=L-p[ids,dim]-h[ids] if upper else p[ids,dim]-h[ids]
                    add(cc,vv,bb)
        k=np.arange(3*n);u=3*n+1+k
        for sign in [-1.,1.]:add(np.column_stack((k,u)),np.tile([sign,-1.],(3*n,1)),np.zeros(3*n))
        A=coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(offset,6*n+1)).tocsr();rhs=np.concatenate(bounds)
        objective=np.zeros(6*n+1);objective[3*n]=1;objective[3*n+1:]=regularization
        trust=np.tile([step,step,step*.4],n)
        limits=list(zip(-trust,trust))+[(-step,step)]+[(0,None)]*(3*n)
        fit=linprog(objective,A_ub=A,b_ub=rhs,bounds=limits,method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-9,'time_limit':2})
        if not fit.success:
            step*=.4
            if step<2e-7:break
            continue
        move=fit.x[:3*n].reshape(n,3);winner=None
        for alpha in [1.,.5,.25,.125,.0625,.03125]:
            trial=repair(p+alpha*move)
            if trial is not None and side(trial)<L-2e-10 and (winner is None or side(trial)<side(winner)):winner=trial
        if winner is None:
            step*=.4
            if step<2e-7:break
        else:
            p=winner;trace.append(dict(iteration=it,before=L,after=side(p),trust=step));step=min(.06,step*1.3)
    return p,trace,time.perf_counter()-at
