"""Regenerate v1-cpu-fast2/train_v1.py from v1/train_v1.py with CPU-safe tweaks.

Keeps one source of truth (the GPU kernel file) while the CPU fallback
kernel differs only in: dataloader workers=0 (no worker-OOM), a 6.5h
time budget guard, 512px / bs=2, and MRCNN disabled.
"""
import re
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
    "    ld_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)\n"
    "    ld_val = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)",
    "dataloader workers")

dst = sub_once(dst,
    "        if val_dice > best_val:",
    "        if time.time() - t0 > 23400:\n"
    "            print(\"[unet] time budget 6.5h reached, stopping early\", flush=True)\n"
    "            break\n"
    "        if val_dice > best_val:",
    "time budget guard")

dst = sub_once(dst,
    "    # GPU kernel configuration (Kaggle script CLI args are not delivered via push);\n"
    "    # FILAMENT_SMOKE=1 lets local tests use the CLI args as-is.\n"
    "    if not os.environ.get(\"FILAMENT_SMOKE\"):\n"
    "        args.unet_epochs = 12\n"
    "        args.mrcnn_epochs = 10\n"
    "        args.size = 1024",
    "    # CPU kernel configuration (Kaggle script CLI args are not delivered via push);\n"
    "    # FILAMENT_SMOKE=1 lets local tests use the CLI args as-is.\n"
    "    if not os.environ.get(\"FILAMENT_SMOKE\"):\n"
    "        args.unet_epochs = 10\n"
    "        args.mrcnn_epochs = 0\n"
    "        args.size = 512\n"
    "        args.unet_bs = 2",
    "cpu main overrides")

dst = sub_once(dst,
    "        unet, UNET_FG_DICE = train_unet(train, val, args.unet_epochs, bs=args.unet_bs, size=args.size, device=device)",
    "        unet, UNET_FG_DICE = train_unet(train, val, args.unet_epochs, bs=args.unet_bs, size=args.size, device=device)",
    "unet train call (anchor)")

out = RUN / "src/notebooks/v1-cpu-fast2/train_v1.py"
out.write_text(dst)
print(f"wrote {out} ({len(dst.splitlines())} lines)")
