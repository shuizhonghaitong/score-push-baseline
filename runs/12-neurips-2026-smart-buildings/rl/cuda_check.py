import torch
assert torch.cuda.is_available(), "CUDA unavailable; aborting as required"
print("cuda:", torch.cuda.get_device_name(0), flush=True)
