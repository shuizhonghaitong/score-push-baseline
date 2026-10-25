#!/bin/bash
set -u
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/18-climx-main-track
BASE="https://hf-mirror.com/datasets/isp-uv-es/ClimX/resolve/main/full"
cd "$RUN/data"
dl() { bash "$RUN/scripts/fetch_robust.sh" "$BASE/$1" "$2" "$1"; }
dl NorESM2-MM_r1i1p1f1_with_forcing_test_forcings_ssp245.nc 1376353869
dl NorESM2-MM_r1i1p1f1_with_forcing_train_ssp126.nc 33149273169
dl NorESM2-MM_r1i1p1f1_with_forcing_train_ssp370.nc 33122625962
dl NorESM2-MM_r1i1p1f1_with_forcing_train_ssp585.nc 33108711240
dl NorESM2-MM_r1i1p1f1_with_forcing_hist_train_1850_1931.nc 31580665406
dl NorESM2-MM_r1i1p1f1_with_forcing_hist_train_1932_2014.nc 31935295647
dl NorESM2-MM_r1i1p1f1_with_forcing_forcings_all.nc 6466332204
echo "ALL FULL FILES DONE"; ls -la
