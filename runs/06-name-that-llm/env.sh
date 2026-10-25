export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
P=http://172.28.4.16:18999
export HTTP_PROXY=$P HTTPS_PROXY=$P http_proxy=$P https_proxy=$P proxy=$P ftp_proxy=$P
export NO_PROXY=localhost,127.0.0.1,::1,172.16.0.0/12,10.0.0.0/8,192.168.0.0/16,100.64.0.0/10,.4pd.io,phanthy.com,nexus.4pd.io,gitlab.4pd.io,contest.4pd
export no_proxy=$NO_PROXY
export CUDA_VISIBLE_DEVICES=
export HF_ENDPOINT=https://hf-mirror.com
