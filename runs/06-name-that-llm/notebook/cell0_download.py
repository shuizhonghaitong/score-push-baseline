# Download the public pools once into ./extra (internet on, Kaggle env creds)
import os, shutil, time
T0=time.time()
os.makedirs("extra", exist_ok=True)
import kagglehub
d1 = kagglehub.dataset_download("ahmedberatozer/ntl-pool-v4")
d2 = kagglehub.dataset_download("prvsiyan/prvsiyans-extended-name-that-llm-dataset")
for src, dst in [(os.path.join(d1, "ntl_pool_v4.parquet"), "extra/ntl_pool_v4.parquet"),
                 (os.path.join(d2, "passages.csv"), "extra/passages.csv")]:
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)
    print(dst, os.path.getsize(dst) if os.path.exists(dst) else "MISSING")
print(f"downloads {time.time()-T0:.0f}s")
