import os, json, shutil, subprocess, sys
os.environ["KAGGLE_API_TOKEN"] = os.environ.get("KAGGLE_API_V1_TOKEN", "")
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle", "kagglehub"], capture_output=True, text=True, timeout=300)
print("pip rc=", r.returncode)
os.makedirs("/kaggle/output", exist_ok=True)
with open("/kaggle/output/probe.txt", "w") as f:
    f.write("hello v5\n")
up = "/kaggle/upload"
os.makedirs(up, exist_ok=True)
shutil.copytree("/kaggle/output", up, dirs_exist_ok=True)
json.dump({"title": "miscc out probe", "id": "lilac1996/miscc-out-probe", "licenses": [{"name": "CC0-1.0"}]},
          open(os.path.join(up, "dataset-metadata.json"), "w"))
r = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "create", "-p", up, "-q"], capture_output=True, text=True, timeout=300)
print("rc=", r.returncode)
print((r.stdout + r.stderr).strip()[:600])
