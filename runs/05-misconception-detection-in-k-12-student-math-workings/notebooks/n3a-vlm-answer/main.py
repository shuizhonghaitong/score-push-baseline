import os, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from PIL import Image

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/1e9,1), "GB")

def find_data():
    for c in ["/kaggle/input/competitions/misconception-detection-in-k-12-student-math-workings",
              "/kaggle/input/misconception-detection-in-k-12-student-math-workings", "/kaggle/input"]:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "train.csv")):
            return c
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train.csv" in files:
            return root
    raise SystemExit("no data")

DATA = find_data()
IMG = os.path.join(DATA, "competition_data", "images") if os.path.isdir(os.path.join(DATA, "competition_data")) else os.path.join(DATA, "images")
print("DATA =", DATA)
def upload_dataset_files(files, ds_slug):
    import json, shutil, subprocess, sys
    os.environ["KAGGLE_API_TOKEN"] = os.environ.get("KAGGLE_API_V1_TOKEN", "")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle", "kagglehub"],
                   capture_output=True, text=True, timeout=600)
    up = "/kaggle/upload_ds"
    shutil.rmtree(up, ignore_errors=True)
    os.makedirs(up, exist_ok=True)
    for f in files:
        shutil.copy(f, os.path.join(up, os.path.basename(f)))
    json.dump({"title": ds_slug, "id": f"lilac1996/{ds_slug}", "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(up, "dataset-metadata.json"), "w"))
    r = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "create", "-p", up, "-q"],
                       capture_output=True, text=True, timeout=1200)
    print("DATASET", ds_slug, "rc=", r.returncode, (r.stdout + r.stderr).strip()[:300], flush=True)
    print(f"DATASET_TAG {ds_slug} lilac1996/{ds_slug}", flush=True)
    return r.returncode == 0


tr = pd.read_csv(os.path.join(DATA, "train.csv"))
te = pd.read_csv(os.path.join(DATA, "test.csv"))

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
t0 = time.time()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16).to(device).eval()
proc = AutoProcessor.from_pretrained(MODEL)
print("model loaded", time.time()-t0, "mem", round(torch.cuda.memory_allocated()/1e9, 1), "GB")

PROMPT = """You are looking at a photo of a student's handwritten math working on a practice sheet.
Question (as shown to the student): @Q@
The correct answer key is: @A@

Task: Look at the student's handwritten working. Determine the student's FINAL ANSWER (the value the student concludes with).
1. Is the student's final answer equivalent to the answer key? (mathematical equivalence, different notation allowed)
2. Reply with ONE line only, exactly in this format:
FINAL_ANSWER=<student answer> MATCH=<yes or no>
If you cannot read a final answer, write FINAL_ANSWER=UNREADABLE MATCH=unknown
"""

def check(img_path, qtext, ans):
    try:
        im = Image.open(img_path).convert("RGB")
    except Exception as e:
        return "UNREADABLE", "unknown", f"load_error:{e}"
    # downscale to limit tokens (max side 960)
    w, h = im.size
    m = max(w, h)
    if m > 960:
        sc = 960 / m
        im = im.resize((int(w*sc), int(h*sc)))
    text = PROMPT.replace("@Q@", qtext[:600]).replace("@A@", str(ans))
    messages = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": text}]}]
    inp = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inp = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inp.items()}
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=60, do_sample=False)
    gen = out[0][inp["input_ids"].shape[1]:]
    s = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
    fa = "UNREADABLE"; match = "unknown"
    for line in s.splitlines():
        if "FINAL_ANSWER=" in line and "MATCH=" in line:
            fa = line.split("FINAL_ANSWER=")[1].split("MATCH=")[0].strip()
            raw_match = line.split("MATCH=")[1].strip().lower()
            first = "".join(ch for ch in (raw_match.split()[0] if raw_match.split() else "") if ch.isalnum())
            if first in ("yes", "y", "correct", "true", "equivalent"):
                match = "yes"
            elif first in ("no", "n", "wrong", "incorrect", "false"):
                match = "no"
            else:
                match = "unknown"
            break
    return fa, match, s

def run_split(df, split):
    rows = []
    t0 = time.time()
    for i, (_, r) in enumerate(df.iterrows()):
        p = os.path.join(IMG, split, r.image_file)
        fa, match, raw = check(p, r.question_text, r.correct_answer)
        rows.append({"trace_id": r.trace_id, "vlm_final_answer": fa, "vlm_match": match})
        if (i+1) % 100 == 0:
            dt = time.time()-t0
            print(f"{split} {i+1}/{len(df)} {dt:.0f}s rate={(i+1)/max(dt,1e-6):.2f}/s", flush=True)
    return pd.DataFrame(rows)

res_tr = run_split(tr, "train")
res_te = run_split(te, "test")
os.makedirs("/kaggle/output", exist_ok=True)
res_tr.to_csv("/kaggle/output/vlm_answer_train.csv", index=False)
res_te.to_csv("/kaggle/output/vlm_answer_test.csv", index=False)
upload_dataset_files(["/kaggle/output/vlm_answer_train.csv", "/kaggle/output/vlm_answer_test.csv"], "miscc-vlm-ans-v1")
print("match distribution train:", res_tr.vlm_match.value_counts().to_dict())
# sanity: compare with gold final_answer_correct on train
gold = tr.set_index("trace_id").final_answer_correct
comp = res_tr.set_index("trace_id")
agree = ((comp.vlm_match == "yes") == gold).mean()
print("train agreement (match==yes vs gold final_answer_correct):", round(agree, 4))
print("DONE")
