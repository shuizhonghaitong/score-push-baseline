"""Generate local run variants from the kernel sources.

Env patches (all variants): TFB_WORK_ROOT, TFB_THREADS.
Usage: .venv/bin/python local/prepare_variants.py
"""
import ast
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]

ENV_P = [
    ("WORK = Path('/kaggle/working')",
     "WORK = Path(os.environ.get('TFB_WORK_ROOT', '/kaggle/working'))"),
    ("THREADS = min(4, os.cpu_count() or 1)",
     "THREADS = int(os.environ.get('TFB_THREADS', 4))"),
    ("os.environ.update(TFB_DATA_ROOT=str(DATA), TFB_OUTPUT_ROOT=str(OUTPUT), TFB_THREADS=str(min(4, os.cpu_count() or 1)))",
     "os.environ.update(TFB_DATA_ROOT=str(DATA), TFB_OUTPUT_ROOT=str(OUTPUT), TFB_THREADS=str(THREADS))"),
]

VARIANTS = [
    # name, source, extra replacements, description
    ("va", "nb_v1/run_v1.py", [], "v1 verbatim (onset .55 gpu .5, ongoing .40)"),
    ("vb", "nb_v1/run_v1.py", [("probability >= 0.4000000000000001", "probability >= 0.45")],
     "v1 + ongoing 0.45"),
    ("vc", "nb_v5e/run_v5e.py.disabled", [("USE_T1_EXTRA = True", "USE_T1_EXTRA = False")],
     "v1 state + onset boost + ongoing 0.45 (no t1)"),
    ("vd", "nb_v4d2/run_v4d2.py.disabled", [], "v4d2 full: t1extra + ongoing 0.45 (needs torch)"),
    ("ve", "nb_v5e/run_v5e.py.disabled", [], "v5e full: t1extra + onset boost + ongoing 0.45 (needs torch)"),
]

for name, src_rel, extra, desc in VARIANTS:
    src = (RUN / src_rel).read_text()
    for old, new in ENV_P + extra:
        n = src.count(old)
        assert n >= 1, (name, old[:60], n)
        src = src.replace(old, new)
    ast.parse(src)
    dst_dir = RUN / "local" / f"nb_{name}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"run_{name}.py"
    dst.write_text(src)
    print(f"local/nb_{name}/run_{name}.py  <- {src_rel}  [{desc}]")
print("all variants written")
