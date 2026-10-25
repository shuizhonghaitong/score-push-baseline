"""Batched seam-kick SAT pressure search on PyTorch CPU / CUDA / Apple MPS.

Float32 tensors propose layouts; CPU float64 SAT repair and trust-region LP
admit candidates. Unit-square side lengths are constant throughout. No tensor
loss or float32 feasibility result is used as a competition validity test.

Inspired by established soft packing and contact continuation. This independent
portable implementation follows the inherited geometry/contact-search lineage.
"""
import time,math
import numpy as np
from scipy.spatial import cKDTree
from packing_geometry import normalize,side,repair

def choose_device(backend):
    import torch
    if backend not in ('cpu','cuda','mps'):raise ValueError('Choose cpu, cuda or mps explicitly.')
    # Do not probe or initialize any accelerator in CPU mode.
    if backend=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Enable a Kaggle NVIDIA GPU or choose cpu.')
    if backend=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable. MPS needs Apple Silicon/macOS; Kaggle uses cpu or cuda.')
    return torch.device(backend)

def tensor_loss(centres,angles,pairs,box):
    import torch
    i,j=pairs[:,0],pairs[:,1];a=angles[:,i];b=angles[:,j]
    dx=centres[:,j,0]-centres[:,i,0];dy=centres[:,j,1]-centres[:,i,1]
    projections=torch.stack((dx*torch.cos(a)+dy*torch.sin(a),-dx*torch.sin(a)+dy*torch.cos(a),dx*torch.cos(b)+dy*torch.sin(b),-dx*torch.sin(b)+dy*torch.cos(b)),dim=-1)
    radius=(1+torch.abs(torch.cos(a-b))+torch.abs(torch.sin(a-b)))*.5
    penetration=torch.relu(radius-torch.amax(torch.abs(projections),dim=-1))
    h=(torch.abs(torch.cos(angles))+torch.abs(torch.sin(angles)))*.5
    low=torch.relu(h[:,:,None]-centres);high=torch.relu(centres+h[:,:,None]-box)
    return penetration.square().sum(1)+low.square().sum((1,2))+high.square().sum((1,2))

def seam_pressure(p,backend='cpu',seed=0,batch=6,steps=240,shrink=.001,seconds=20):
    import torch
    device=choose_device(backend)
    if backend=='cpu':torch.set_num_threads(2)
    p=normalize(p);n=len(p);L=side(p);rng=np.random.default_rng(seed);start=time.perf_counter()
    pair_array=cKDTree(p[:,:2]).query_pairs(2.2,output_type='ndarray')
    # Centre moves are projected to +/-0.25 per coordinate. An omitted pair
    # starts farther than 2.2 > sqrt(2)+2sqrt(2)*0.25 and therefore cannot meet,
    # even if both squares rotate. Final admission independently checks all pairs.
    pairs=torch.tensor(pair_array,dtype=torch.long,device=device)
    original=torch.tensor(p[:,:2],dtype=torch.float32,device=device)
    original_angles=torch.tensor(p[:,2],dtype=torch.float32,device=device)
    initial=np.repeat(p[None,:,:2],batch,axis=0);initial_angles=np.repeat(p[None,:,2],batch,axis=0)
    box=L-shrink
    for k in range(batch):
        # Opposed tangential motion across one horizontal/vertical seam; a
        # coherent kick can explore different contacts from independent jitter.
        axis=k%2;cut=rng.uniform(.2,.8)*L;halves=np.where(p[:,axis]>cut,1.,-1.)
        initial[k,:,1-axis]+=halves*rng.choice([-1.,1.])*rng.uniform(.0002,.012)
        initial_angles[k]+=rng.normal(0,.0004,n)
    initial[:,:,:2]=(initial[:,:,:2]-L*.5)*(box-1)/(L-1)+box*.5
    centres=torch.tensor(initial,dtype=torch.float32,device=device,requires_grad=True)
    angles=torch.tensor(initial_angles,dtype=torch.float32,device=device,requires_grad=True)
    optimizer=torch.optim.Adam([{'params':[centres],'lr':.002},{'params':[angles],'lr':.00012}])
    best_loss=torch.full((batch,),float('inf'),device=device);best_c=centres.detach().clone();best_a=angles.detach().clone();trace=[]
    completed=0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True);loss=tensor_loss(centres,angles,pairs,box)
        if not bool(torch.isfinite(loss).all().item()):raise FloatingPointError('Non-finite pressure loss')
        with torch.no_grad():
            better=loss<best_loss;best_loss=torch.where(better,loss,best_loss)
            best_c=torch.where(better[:,None,None],centres.detach(),best_c)
            best_a=torch.where(better[:,None],angles.detach(),best_a)
        loss.sum().backward();optimizer.step()
        with torch.no_grad():
            centres.copy_(torch.maximum(torch.minimum(centres,original+.25),original-.25))
            angles.copy_(torch.maximum(torch.minimum(angles,original_angles+.05),original_angles-.05))
        completed=step+1
        if step%40==0:
            trace.append({'step':step,'loss':float(best_loss.min().item())})
            if time.perf_counter()-start>seconds:break
        if step in [steps//2,3*steps//4]:
            for group in optimizer.param_groups:group['lr']*=.25
    # Transfer only tiny candidate arrays, not autograd state, to CPU.
    cc=best_c.detach().cpu().numpy().astype(float);aa=best_a.detach().cpu().numpy().astype(float)
    proposals=[]
    for k in range(batch):
        q=repair(np.column_stack((cc[k],aa[k])),clearance=2e-9)
        if q is not None:proposals.append(q)
    proposals.sort(key=side)
    report=dict(backend=backend,batch=batch,steps=completed,seconds=time.perf_counter()-start,trace=trace,proposal_sides=[side(q)for q in proposals],before=L)
    return proposals,report

if __name__=='__main__':
    import argparse,json
    from packing_geometry import load_bank,save_bank
    from cpu_contact_polish import polish
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--backend',choices=['cpu','cuda','mps'],default='cpu');parser.add_argument('--sizes',nargs='+',type=int,default=[984,553]);parser.add_argument('--steps',type=int,default=240);parser.add_argument('--batch',type=int,default=6);parser.add_argument('--prefix',default='portable_seam')
    args=parser.parse_args();bank=load_bank(args.input);history=[]
    for n in args.sizes:
        initial=side(bank[n]);proposals,report=seam_pressure(bank[n],backend=args.backend,seed=81173+n,batch=args.batch,steps=args.steps)
        for q in proposals[:2]:
            if side(q)>initial+.015:continue
            q,_,_=polish(q,rounds=50,max_seconds=5,trust_start=.015)
            if side(q)<side(bank[n])-1e-8:bank[n]=q
        report.update(n=n,after=side(bank[n]),gain=(initial-side(bank[n]))/math.sqrt(n));history.append(report);print(json.dumps(report),flush=True)
    save_bank(args.prefix+'_packings.npz',bank)
    from pathlib import Path
    Path(args.prefix+'_history.json').write_text(json.dumps(history,indent=2))
