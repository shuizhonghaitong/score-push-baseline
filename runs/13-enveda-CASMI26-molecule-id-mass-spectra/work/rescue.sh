#!/bin/bash
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999
export http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
exec /mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python work/rescue_push.py
