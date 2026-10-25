#!/bin/bash
set -u
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/18-climx-main-track
BASE="https://hf-mirror.com/datasets/isp-uv-es/ClimX/resolve/main/lite"
cd "$RUN/data"
dl() { bash "$RUN/scripts/fetch_robust.sh" "$BASE/$1" "$2" "$1"; }
dl NorESM2-MM_r1i1p1f1_with_forcing_subsampled_16_hist_train_1850_1931.nc 130742102
dl NorESM2-MM_r1i1p1f1_with_forcing_subsampled_16_hist_train_1932_2014.nc 132229080
dl NorESM2-MM_r1i1p1f1_with_forcing_subsampled_16_test_forcings_ssp245.nc 6263403
dl NorESM2-MM_r1i1p1f1_with_forcing_subsampled_16_train_ssp126.nc 137521078
dl NorESM2-MM_r1i1p1f1_with_forcing_subsampled_16_train_ssp370.nc 137385065
dl NorESM2-MM_r1i1p1f1_with_forcing_subsampled_16_train_ssp585.nc 137355114
ls -la
