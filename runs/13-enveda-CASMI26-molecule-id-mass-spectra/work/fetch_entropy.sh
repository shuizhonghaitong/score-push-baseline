#!/bin/bash
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra
kaggle kernels output lilac1996/casmi26-v2-entropy -p work/entropy_out --file-pattern 'submission' 2>&1 | tail -5
find work/entropy_out -type f | head
