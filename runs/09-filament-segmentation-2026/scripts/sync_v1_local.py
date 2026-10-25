"""Regenerate src/notebooks/v1-local/train_v1.py from v1/train_v1.py.

Local CPU safety-net variant (8-core, 2 TB RAM): U-Net only, 768px,
bs=4, workers=4, 12 epochs, no wall-clock budget guard.
Run after editing src/notebooks/v1/train_v1.py.
"""
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
src = (RUN / "src/notebooks/v1/train_v1.py").read_text()
dst = src

def sub_once(text, old, new, tag):
    assert text.count(old) == 1, f"{tag}: expected 1 occurrence, found {text.count(old)}"
    return text.replace(old, new)

dst = sub_once(dst,
    "    ld_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)\n"
    "    ld_val = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)",
    "    ld_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)\n"
    "    ld_val = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=4, pin_memory=False)",
    "dataloader workers")

dst = sub_once(dst,
    "    if not os.environ.get(\"FILAMENT_SMOKE\"):\n"
    "        args.unet_epochs = 12\n"
    "        args.mrcnn_epochs = 10\n"
    "        args.size = 1024",
    "    if not os.environ.get(\"FILAMENT_SMOKE\"):\n"
    "        args.unet_epochs = 12\n"
    "        args.mrcnn_epochs = 0\n"
    "        args.size = 768\n"
    "        args.unet_bs = 4",
    "local main overrides")

out = RUN / "src/notebooks/v1-local"
out.mkdir(parents=True, exist_ok=True)
(out / "train_v1.py").write_text(dst)
print(f"wrote {out / 'train_v1.py'} ({len(dst.splitlines())} lines)")
