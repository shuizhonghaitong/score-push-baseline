#!/bin/bash
# Stage-4: publish both scored outputs as private datasets, push ensemble notebook, submit.
# Run manually after BOTH scored reruns have outputs.
set -x
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra

# 1) pull outputs (fetch_artifacts.sh / fetch_entropy.sh must have run first)
mkdir -p ds_v465 ds_entropy
V465=$(find work/v465_out -name 'submission.csv' | head -1)
ENT=$(find work/entropy_out -name 'submission.csv' | head -1)
[ -f "$V465" ] && cp "$V465" ds_v465/submission_v465.csv
[ -f "$ENT" ] && cp "$ENT" ds_entropy/submission_entropy.csv
cat > ds_v465/dataset-metadata.json <<'M1'
{"title": "CASMI26 V465 out", "id": "lilac1996/casmi26-v465-out", "is_private": true,
 "subjects": ["Science"], "keywords": ["casmi26", "internal"],
 "licenses": [{"name": "MIT"}]}
M1
cat > ds_entropy/dataset-metadata.json <<'M2'
{"title": "CASMI26 entropy out", "id": "lilac1996/casmi26-entropy-out", "is_private": true,
 "subjects": ["Science"], "keywords": ["casmi26", "internal"],
 "licenses": [{"name": "MIT"}]}
M2
kaggle datasets create -p ds_v465
kaggle datasets create -p ds_entropy

# 2) attach datasets to the ensemble notebook and push (429-aware: run only after a known-clear window)
sed -i 's/"dataset_sources": \[\]/"dataset_sources": ["lilac1996\/casmi26-v465-out", "lilac1996\/casmi26-entropy-out"]/' work/nb_ens/kernel-metadata.json
kaggle kernels push -p work/nb_ens

# 3) submit the ensemble kernel (CPU rerun on hidden test)
kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra \
  -k lilac1996/casmi26-ensemble-rrf \
  -m "ensemble RRF: V465 fork + backup entropy engine (autonomous run 13)"
