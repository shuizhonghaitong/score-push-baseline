import os, time, json, warnings, re
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

def find_data():
    for c in ["/kaggle/input/misconception-detection-in-k-12-student-math-workings", "/kaggle/input"]:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "train.csv")):
            return c
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train.csv" in files:
            return root
    raise SystemExit("no data")

DATA = find_data()
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
codes = sorted(tr.root_code.unique())
CODELIST = ", ".join(codes)
CODE_DESC = {"EXP-01": "misinterpreted 2^-7 as either 2 mm or 7 mm. However, 2^\u22127means 1/128 not 2 or 7", "NOT-01": "it shoulde be a^(2+1)/2", "INT-04": "Incorrect calculation for -9 + 1. It should be -8.", "ALG-06": "After obtaining 11a=7, the coefficient should be divided by 11 not by subtracting.", "ANS-01": "it shoulde be a^(3/2 \u00d7 1/3) =a^1/2", "ALG-04": "The value of 6(x+1) is incorrect. It should be 6x+6 instead of 6x+1", "FUN-01": "The student made a conceptual error by substituting f(5) = 5 - 1, even though this information is not provided anywhere in the problem....", "EXP-03": "The value of (2a\u22121)^2 is incorrect, it should be 4a^2 - 4a + 1, instead of 4a^2 + 1.", "ALG-02": "The unlike terms 8x and 1 are conjoined into 9x. A variable term and a constant cannot be added into a single term; 8x + 1 cannot be...", "EXP-02": "Instead of immediate substitution of 5 into 3^2a,we simplify the product first using exponent rules; since the bases are the same (base...", "WOP-01": "The correct operation to use is subtraction, not addition.", "CAL-01": "It should be C(6,2).", "FRA-01": "The student made a mistake when calculating with fractions.", "OTH-00": "The student used an incorrect operation and failed to simplify/calculate the final fraction result", "ANG-02": "The three sides lengths were added and set equal to 180, but 180 is the sum of the angles of a triangle. Since this is an equilateral...", "ANS-02": "It can not directly state that the function's value is 0. No need to find the value of a, just write the function of f(2a-1) in the form...", "EXP-05": "The result of 2^(3-3) = 2^0 is not 2. it should be 2^0 = 1, so the denominator should be 1.", "INT-03": "Subtracting a negative should be converted to addition: 100 - (-20) = 100 + 20 = 120, not -5.", "EXP-04": "the square root should not be removed", "INT-02": "the calculation result of 12 x (-3) =36 is incorrect. it should be -36", "OPS-01": "The first column subtraction is incorrect: 85.000 \u2212 34.850 = 50.150, not 54.150. This is a borrowing/column-subtraction error.", "OOO-02": "34650 in L1 should be -34650. Then, -34650+12800=-21850. the last step is 90.000-21850=68150. so the answer is 68150.", "EXP-07": "The student's answer is not fully simplified; it should be 3x\u221a5x", "NOT-02": "Chained equalities reuse the '=' sign: '1\u00d72=2\u00d75=10' literally states 1\u00d72 = 2\u00d75, which is false (2 \u2260 10). The steps should be written...", "OPS-02": "division error and the final calculating is incomplete", "MEA-02": "incorrect time conversion. 288 menit = 4 jam 48 menit not 4 jam 8 menit", "FRM-01": "incorrect factors selection for LCM", "KPF-04": "Jawaban: 5\u00d72=10 in L4 should be Jawaban: 5\u00d72x2=20.", "KPF-01": "The student misunderstood the problem by calculating both the LCM and GCF, even though only the GCF was required.", "MEA-05": "No 3 karena luas bukan banyak kalau banyak in L12 should be no.3 and no.1 because surface area is the total area of all the flat or...", "MEA-01": "The student's answer is incorrect; the formula for the perimeter of a rectangle should be $2 \\times (p + l)$.", "CHK-01": "The final probability 30/11 is greater than 1, which is impossible for a probability. The result should be 5/66.", "INV-01": "The student solved f(x) = 2, but f\u207b\u00b9(x) = 2 means f(2) = x. Therefore, the required value is x = f(2) = 1/(2 \u2212 4) = \u22121/2.", "WOP-02": "It should be 90/3 =30", "FRA-05": "The mixed number 2 2/5 was converted incorrectly as 7/5. The correct conversion is (2 \u00d7 5 + 2)/5 = 12/5.", "Dec-02": "The decimal 0.225 has three digits after the decimal comma, so it must first be written as 225/1000, not 225/100. The correct conversion...", "REL-01": "Not a solution to the question asked.", "KPF-03": "Students are unable to correctly determine KPK and correct the final answer.", "KPF-02": "the student mapped \"3rd meeting\" directly to multiplying by 3 (3*80), failing to account for the initial starting point (day 0).", "FRA-04": "The student simplified 12/5 to 4/2, which is not an equivalent fraction because the numerator and denominator were not divided by the...", "VIS-01": "The student ignored the position of m on the left side of 0 in the number line. A point 6 units to the left of 0 has value m = -6, not m...", "LOG-02": "It should be [(-2).(-2) (^p log p )]/2 = 4/4 = 1.", "FRA-06": "It should be 5/2 \u00d7 4/10 = 20/20 = 1, so 1 - 1/3 = 3/3 - 1/3 = 2/3", "QUA-02": "The factors of 2x^2-3x-5 is incorrect, it should be (2x-5)(x+1) and no need to used determinane to get the factors.", "FRA-03": "The final result \u22121/8 is incorrect. After simplifying the two logarithms in the numerator, their product is 4, while log_r(r\u00b2) = 2....", "OOO-01": "The final expression is not evaluated correctly because the numerator was not combined before division. The correct calculation is (3/2...", "UNT-01": "The student adds 15 minutes to 6\u00b0C, mixing incompatible quantities. The elapsed time must first be converted into the number of 3-minute...", "MEA-04": "The area formula for a rhombus is incomplete. The diagonals must be multiplied and then divided by 2: (1/2) \u00d7 24 \u00d7 18 = 216 cm\u00b2, not 24...", "ALG-01": "When simplifying \u221a(200x\u00b2), the student dropped the variable x\u00b2 from under the radical and wrote only \u221a(100\u00d72). In L3-L4 the student...", "ALG-03": "The student combined unlike terms: 10 \u2212 4\u221a5 \u2212 5\u221a3 + 2\u221a15 was forced into 3\u221a17 (coefficients summed 10\u22124\u22125+2 = 3 and radicands merged...", "ALG-05": "The equation was transformed incorrectly by separating and cancelling parts of the numerator and denominator instead of multiplying both...", "STA-01": "The mean formula is misapplied: the weighted total must be divided by the total frequency. The correct equation is (530 + 60x)/(10 + x)...", "ANG-01": "The two given angles are corresponding angles formed by parallel rails, so they are equal, not supplementary. The correct equation is...", "SIM-01": "The elimination result \u22122b = \u221214 was divided incorrectly: dividing both sides by \u22122 gives b = 7, not b = \u22127. The correct common...", "MEA-03": "The cube's volume is labeled in cm rather than cm\u00b3. Volume should use cubic units.", "COR-01": "Reflecting point B(3,2) across the line y = -x should give C(-2,-3), because the reflection rule over y = -x is (x,y) -> (-y,-x). The...", "Dec-01": "The student judged 0,308 to be greater than 0,35, apparently because 0,308 has more digits (or because 308 > 35 as whole numbers)....", "EXP-06": "After obtaining 9 = 3^2, the student should equate the exponents and conclude x = 2. Instead, the student wrote x = 3^2, treating the...", "FRA-02": "All three decimal-to-fraction conversions are correct (Leo = 12 11/20, Adam = 12 1/20, Doni = 12 1/2), but the final ranking is wrong....", "PRO-03": "The value 60 is left standing as the answer to a probability question. A probability can never exceed 1, so 60 should have been...", "PRO-04": "For the two DIFFERENT front letters the student counts 5 \u00d7 5 = 25, ignoring the restriction that the letters must be different. Once the...", "GEO-01": "The student called the cylinder face that coincides with the hemisphere the 'base circle of the cylinder', whereas the coinciding face...", "INT-01": "The student ordered the cities from coldest to warmest as S, T, Q, P, R, i.e., by the unsigned magnitude of the numbers (0, 1, 3, 4, 9)....", "LIN-01": "Inconsistent point order between numerator and denominator: the y-side uses 40\u221225 (second point minus first), but the x-side denominator...", "STA-05": "The student treats the quartile position 3.75 as an interpolated value of 2.75. Under the verified convention for this question, the...", "RAT-02": "The student treats 15% as the plain number 15. The percentage must be converted to 15/100, so the calculation should be 15/100 \u00d7...", "PRO-01": "For the joint ('and') event, the student ADDS the counts of red and blue pencils (15 + 5 = 20) instead of multiplying the sequential...", "PRO-02": "On the second draw without replacement, the remaining cards should be 12 \u2212 1 = 11, so P(B | first vowel) = 2/11. The student wrote the...", "PV-01": "The multiplication result ignores the place value of the 1 in 16. The second partial product must be shifted one place, so 4,200 \u00d7 16 =...", "RAT-01": "The student treats the growth factor (1+r) as the DIFFERENCE in capital (14.400.000 \u2212 10.000.000 = 4.400.000) instead of the...", "SEQ-01": "The student treats five hours as a single multiplication factor, 5. Because the amoeba population doubles every hour, the factor after 5...", "SEQ-02": "To find the 10th term the student used the sum-of-first-n-terms formula, S_n = n/2(2a+(n-1)b), whereas the question asks for the nth...", "SET-01": "The student counts the favorable outcomes as 25 + 20 = 45, using all students who like Math (25) and all who like Physics (20). This...", "STA-03": "The question gives the mean as 8, but the student changes the task to finding a median and assumes 2n + 1 = 9. Mean and median are...", "STA-04": "The class width is taken as 5, but the classes 1\u20132, 3\u20134, 5\u20136, \u2026 each span width w = 2 (e.g. boundaries 4.5 to 6.5). This wrong width is..."}
TAXONOMY = "\n".join(f"{c}: {CODE_DESC.get(c, '')}" for c in codes if c != "NONE")
IMG = os.path.join(DATA, "competition_data", "images") if os.path.isdir(os.path.join(DATA, "competition_data")) else os.path.join(DATA, "images")

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
t0 = time.time()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16).to(device).eval()
proc = AutoProcessor.from_pretrained(MODEL)
print("model loaded", time.time()-t0, "mem", round(torch.cuda.memory_allocated()/1e9, 1), "GB", flush=True)

PROMPT = """Photo: a student's handwritten working on an Indonesian math practice sheet.
Question (as shown to the student): @Q@
Correct answer key: @A@

Diagnose the student's work. If the work is essentially correct, the root code is NONE.
Otherwise pick the most relevant root codes from this taxonomy:
@CODES@

Reply with ONE line, exactly:
TOP3=<code1> <code2> <code3>
(code1 = most likely; use NONE if appropriate)
"""

CODE_SET = set(codes) | {"NONE"}

def diagnose(img_path, qtext, ans):
    try:
        im = Image.open(img_path).convert("RGB")
    except Exception as e:
        return ["NONE", "ANS-01", "WOP-01"], f"load_error:{e}"
    w, h = im.size
    m = max(w, h)
    if m > 960:
        sc = 960 / m
        im = im.resize((int(w*sc), int(h*sc)))
    text = PROMPT.replace("@Q@", qtext[:600]).replace("@A@", str(ans)).replace("@CODES@", TAXONOMY)
    messages = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": text}]}]
    inp = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inp = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inp.items()}
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=40, do_sample=False)
    s = proc.batch_decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    got = []
    for tok in re.findall(r"[A-Z]+-\d{2}|NONE", s):
        if tok in CODE_SET and tok not in got:
            got.append(tok)
    while len(got) < 3:
        got.append("NONE")
    return got[:3], s

def run_split(df, split):
    rows = []
    t0 = time.time()
    for i, (_, r) in enumerate(df.iterrows()):
        p = os.path.join(IMG, split, r.image_file)
        got, raw = diagnose(p, r.question_text, r.correct_answer)
        rows.append({"trace_id": r.trace_id, "p1": got[0], "p2": got[1], "p3": got[2], "raw": raw[:200]})
        if (i+1) % 100 == 0:
            dt = time.time()-t0
            print(f"{split} {i+1}/{len(df)} {dt:.0f}s rate={(i+1)/max(dt,1e-6):.2f}/s", flush=True)
    return pd.DataFrame(rows)

res_tr = run_split(tr, "train")
res_te = run_split(te, "test")

def map3(df, gold):
    tot = 0.0
    for (_, r), yi in zip(df.iterrows(), gold):
        seen = set(); sc = 0.0
        for k, p in enumerate([r.p1, r.p2, r.p3], 1):
            if p in seen: continue
            seen.add(p)
            if p == yi:
                sc = 1.0/k; break
        tot += sc
    return tot/len(df)

print("train OOF MAP@3 (zero-shot diag):", round(map3(res_tr, tr.root_code.to_numpy()), 4), flush=True)
os.makedirs("/kaggle/output", exist_ok=True)
res_tr.to_csv("/kaggle/output/vlm_diag_train.csv", index=False)
res_te.to_csv("/kaggle/output/vlm_diag_test.csv", index=False)
upload_dataset_files(["/kaggle/output/vlm_diag_train.csv", "/kaggle/output/vlm_diag_test.csv"], "miscc-vlm-diag-v1")
print("p1 distribution train:", res_tr.p1.value_counts().head(10).to_dict())
print("DONE")
