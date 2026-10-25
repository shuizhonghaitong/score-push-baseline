"""Per-photo feature extraction: mm-normalized physical features."""
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

_SPEC_EDGES = np.linspace(-1.0, 3.5, 25)   # log10 period bins, ~0.1mm..3000mm (24 bins)
_EDGE_MM = [0.2, 0.63, 2.0, 6.3, 20.0]
_LV_MM = [0.5, 2.0, 8.0]


def _odd_kernel(k, dim):
    k = int(round(k))
    if k % 2 == 0:
        k += 1
    maxk = dim if dim % 2 == 1 else dim - 1
    return max(1, min(k, maxk))


def load_gray(path, target_long=1600, clahe=True):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = target_long / max(h, w)
    if abs(scale - 1.0) > 1e-3:
        tw, th = max(1, int(w*scale)), max(1, int(h*scale))
        cur = img
        # two-step area downscale is far faster for large reductions
        while max(cur.shape[0], cur.shape[1]) > max(tw, th) * 1.5:
            nh, nw = cur.shape[:2]
            cur = cv2.resize(cur, (max(1, nw//2), max(1, nh//2)), interpolation=cv2.INTER_AREA)
        img = cv2.resize(cur, (tw, th), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if clahe:
        gray_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray.astype(np.uint8)).astype(np.float32)
    else:
        gray_eq = gray.copy()
    return img, gray, gray_eq, scale


def fft_grain_spectrum(gray, eff_ppm):
    g = gray - gray.mean()
    win = (np.hanning(gray.shape[0])[:, None] * np.hanning(gray.shape[1])[None, :]).astype(np.float32)
    F = np.fft.fft2(g * win)
    P = np.fft.fftshift(np.abs(F) ** 2)
    H, W = P.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    maxr = min(H, W) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        period_mm = np.where(r > 0.5, (maxr / np.maximum(r, 0.5)) / max(eff_ppm, 1e-6), 1e9)
    logp = np.log10(np.clip(period_mm, 1e-4, 1e4))
    pw = P.ravel(); lp = logp.ravel()
    nbin = len(_SPEC_EDGES) - 1
    idx = np.clip(np.digitize(lp, _SPEC_EDGES) - 1, 0, nbin - 1)
    cnt = np.bincount(idx, minlength=nbin)
    tot = np.bincount(idx, weights=pw, minlength=nbin)
    spec = tot / np.maximum(cnt, 1)
    spec = spec / (spec.sum() + 1e-12)
    return spec


def extract_handcrafted(path, ppm, target_long=1600):
    loaded = load_gray(path, target_long=target_long)
    if loaded is None:
        return None
    img, gray, g, scale = loaded
    H, W = g.shape
    eff_ppm = ppm * scale
    feats = {}
    # ---- FFT grain-size spectrum ----
    spec = fft_grain_spectrum(g, eff_ppm)
    for i, v in enumerate(spec):
        feats[f"spec_{i}"] = float(v)
    centers = 10 ** ((_SPEC_EDGES[:-1] + _SPEC_EDGES[1:]) / 2)
    lc = np.log10(centers)
    dom = centers[np.argmax(spec)]
    w_ = spec / (spec.sum() + 1e-12)
    feats["fft_dom_log_mm"] = float(np.log10(dom))
    feats["fft_log_mean"] = float(np.sum(w_ * lc))
    feats["fft_log_std"] = float(np.sqrt(np.sum(w_ * (lc - feats["fft_log_mean"]) ** 2)))
    feats["spec_frac_coarse_2mm"] = float(np.sum(w_[lc > np.log10(2.0)]))
    feats["spec_frac_coarse_063"] = float(np.sum(w_[lc > np.log10(0.63)]))
    feats["spec_frac_fine_0063"] = float(np.sum(w_[lc < np.log10(0.063)]))
    # ---- edge density at mm scales ----
    base = cv2.equalizeHist(g.astype(np.uint8))
    dim = min(H, W)
    for mm in _EDGE_MM:
        k = _odd_kernel(eff_ppm * mm, dim)
        blurred = cv2.GaussianBlur(base, (k, k), 0)
        feats[f"edge_{mm}mm"] = float(cv2.Canny(blurred, 50, 150).mean() / 255.0)
    # ---- blob diameters (distinct particles) ----
    blur = cv2.GaussianBlur(base, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    area_mm2 = (H * W) / max(eff_ppm ** 2, 1e-6)
    diam = np.sqrt(4.0 * np.maximum(areas, 0) / np.pi) / max(eff_ppm, 1e-6)
    valid = diam[(diam > 0.005) & (diam < 400)]
    if len(valid) > 0:
        for q, name in [(25, "p25"), (50, "p50"), (75, "p75"), (90, "p90"), (99, "p99")]:
            feats[f"blob_{name}_mm"] = float(np.percentile(valid, q))
        feats["blob_mean_mm"] = float(valid.mean())
        feats["blob_n_per_mm2"] = float(len(valid) / max(area_mm2, 1e-6))
        feats["blob_area_frac"] = float(areas.max() / (H * W))
    else:
        for q, name in [(25, "p25"), (50, "p50"), (75, "p75"), (90, "p90"), (99, "p99")]:
            feats[f"blob_{name}_mm"] = 0.0
        feats["blob_mean_mm"] = 0.0
        feats["blob_n_per_mm2"] = 0.0
        feats["blob_area_frac"] = 0.0
    # ---- local roughness at multiple scales ----
    for mm in _LV_MM:
        k = _odd_kernel(eff_ppm * mm, dim)
        m = cv2.blur(g, (k, k)); sqm = cv2.blur(g ** 2, (k, k))
        feats[f"localvar_{mm}mm"] = float(np.sqrt(np.clip(sqm - m ** 2, 0, None).mean()))
    # ---- fine texture proxy ----
    hi = g - cv2.GaussianBlur(g, (7, 7), 0)
    feats["hf_energy"] = float(np.std(hi))
    # ---- LBP ----
    radius = max(1, int(round(eff_ppm * 0.4)))
    lbp = local_binary_pattern(g.astype(np.uint8), 8, radius, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    for i, v in enumerate(hist):
        feats[f"lbp_{i}"] = float(v)
    # ---- GLCM ----
    step = max(1, int(round(eff_ppm * 1.0)))
    glcm = graycomatrix(base, distances=[step], angles=[0, np.pi / 2], levels=256, symmetric=True, normed=True)
    feats["glcm_contrast"] = float(graycoprops(glcm, "contrast").mean())
    feats["glcm_homogeneity"] = float(graycoprops(glcm, "homogeneity").mean())
    feats["glcm_energy"] = float(graycoprops(glcm, "energy").mean())
    feats["glcm_correlation"] = float(graycoprops(glcm, "correlation").mean())
    # ---- intensity / color ----
    bgr = img.astype(np.float32)
    for i, ch in enumerate("BGR"):
        feats[f"int_{ch}_mean"] = float(bgr[:, :, i].mean())
        feats[f"int_{ch}_std"] = float(bgr[:, :, i].std())
    feats["gray_mean"] = float(g.mean())
    feats["gray_std"] = float(g.std())
    m1 = cv2.blur(g, (9, 9)); sqm1 = cv2.blur(g ** 2, (9, 9))
    lv = np.sqrt(np.clip(sqm1 - m1 ** 2, 0, None))
    feats["uniform_frac"] = float((lv < 4.0).mean())
    return feats
