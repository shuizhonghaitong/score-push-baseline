#!/bin/bash
# Stage-3: pull scored V465 fork outputs (after its scoring rerun completes).
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra
kaggle kernels output lilac1996/casmi26-v465-fork -p work/v465_out \
  --file-pattern 'submission|e004_candidate_inputs|lilac_sim_scores|candidate_audit|experiment_audit|run_manifest|ice_out|gl_out|baseline_submission' 2>&1 | tail -5
find work/v465_out -type f | head -30
