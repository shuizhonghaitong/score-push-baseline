import torch
print("cuda available:", torch.cuda.is_available())
n = torch.cuda.device_count()
print("device count:", n)
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name} {p.total_memory/2**30:.1f} GiB sm={p.major}.{p.minor}")
import torch as t
x = t.randn(1024, 1024, device="cuda")
print("matmul ok:", (x @ x).sum().item() != 0)
