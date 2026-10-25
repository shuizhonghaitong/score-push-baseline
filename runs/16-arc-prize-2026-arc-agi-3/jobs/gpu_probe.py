import subprocess
print(subprocess.run(["nvidia-smi","--query-gpu=index,name,memory.total,memory.free","--format=csv"],capture_output=True,text=True).stdout)
import torch
print("cuda:", torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
