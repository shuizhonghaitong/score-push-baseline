"""Shared utilities: metric, parsing, ppm lookup, monotonic post-processing."""
import os, re
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TRAIN_IMG_DIR = f"{DATA_DIR}/Training-All_Photos_updated/Training-All_Photos_updated"
TEST_IMG_DIR = f"{DATA_DIR}/Test_All_Photos/Test_All_Photos"

SUPPORT_MM = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
SUPPORT_COLS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
LOG_X = np.log10(SUPPORT_MM)
LOG_WEIGHTS = np.diff(LOG_X)  # 10 interval widths, sum to 5.0


def emd_trap(y_true, y_pred):
    """Trapezoidal area between cumulative curves on log10(x) axis.
    Per-sample: sum_i (|e_i|+|e_{i+1}|)/2 * w_i. Lower better. Range [0,500]."""
    yt = np.atleast_2d(np.asarray(y_true, float))
    yp = np.atleast_2d(np.asarray(y_pred, float))
    e = np.abs(yt - yp)
    iv = (e[:, :-1] + e[:, 1:]) / 2.0
    return (iv * LOG_WEIGHTS).sum(axis=1)


def emd_trap_mean(y_true, y_pred):
    return float(emd_trap(y_true, y_pred).mean())


def emd_left(y_true, y_pred):
    """Left-Riemann (V4 variant)."""
    yt = np.atleast_2d(np.asarray(y_true, float))
    yp = np.atleast_2d(np.asarray(y_pred, float))
    return (np.abs(yt[:, :-1] - yp[:, :-1]) * LOG_WEIGHTS).sum(axis=1)


def enforce(row):
    """Clip to [0,100], enforce non-decreasing, force 200mm=100."""
    p = np.clip(np.asarray(row, dtype=float), 0, 100)
    p = np.maximum.accumulate(p)
    p[-1] = 100.0
    return p


def enforce_rows(rows):
    return np.array([enforce(r) for r in rows])


# ---------------- filename parsing ----------------
def canon(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def normalize_sample_id(s):
    s = s.replace("Ü", "Ue").replace("ü", "ue").replace("Ä", "Ae").replace("ä", "ae")
    s = s.replace("Ö", "Oe").replace("ö", "oe").replace("ß", "ss").replace(",", "_")
    return s.strip()


def parse_train_filename(path):
    base = os.path.basename(path)
    m = re.search(r"([A-Za-z]\d{3})_(\d+)\.(jpg|jpeg|png)$", base, flags=re.IGNORECASE)
    if not m:
        return None, None
    camera = base[:m.start()].rstrip("_")
    return camera, m.group(1)


def parse_test_filename(path):
    base = os.path.basename(path)
    n = re.sub(r"\.(jpg|jpeg|png)$", "", base, flags=re.IGNORECASE)
    n = re.sub(r"\s*\(\d+\)\s*$", "", n)
    idx = n.find("HPC")
    if idx == -1:
        return "unknown", normalize_sample_id(n)
    camera = n[:idx].rstrip("_").rstrip()
    return camera, normalize_sample_id(n[idx:])


def _canon(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_ppm_table():
    return pd.read_csv(f"{DATA_DIR}/ppm_updated.csv")


def _row_ppm_long(r):
    return float(r["ppm"]), float(max(int(r["width"]), int(r["height"])))


def lookup_ppm(camera_name, ppm_df):
    c = _canon(camera_name)
    rows = ppm_df
    # exact match on phone or camera column
    for _, r in rows.iterrows():
        if c == _canon(r["phone"]) or c == _canon(r["camera"]):
            return _row_ppm_long(r) + (_canon(r["phone"]),)
    # prefix / contained match, pick longest overlap
    best, best_len, best_long = None, -1, None
    for _, r in rows.iterrows():
        for col in ("phone", "camera"):
            rc = _canon(r[col])
            ov = len(c) if c.startswith(rc) else (len(rc) if rc.startswith(c) else 0)
            if ov > best_len and ov > 0:
                best_len, best, best_long = ov, float(r["ppm"]), float(max(int(r["width"]), int(r["height"])))
    if best is not None:
        return best, best_long, "prefix"
    return float(ppm_df["ppm"].mean()), float(ppm_df[["width", "height"]].max(axis=1).mean()), "default"


def list_photos(img_dir, is_train):
    exts = {".jpg", ".jpeg", ".png"}
    paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
             if os.path.splitext(f)[1].lower() in exts]
    paths = sorted(set(paths))
    out = []
    for p in paths:
        if is_train:
            cam, sid = parse_train_filename(p)
            if sid is None:
                print(f"WARNING: unparseable train filename {os.path.basename(p)}")
                continue
        else:
            cam, sid = parse_test_filename(p)
        out.append((p, sid, cam))
    return out
