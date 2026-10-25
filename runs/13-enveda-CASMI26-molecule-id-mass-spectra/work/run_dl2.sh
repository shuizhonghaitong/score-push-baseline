#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999
export http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
TRURL=$(python3 -c "import json;print(json.load(open('/tmp/casmi_urls.json'))['train.parquet'])")
exec ./.venv/bin/python work/paraget.py "$TRURL" data/enveda-casmi26-molecule-id-mass-spectra/train2.parquet 16
