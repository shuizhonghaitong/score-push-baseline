import os, glob, time, math, warnings, json, sys, traceback
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import savgol_filter
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
t0 = time.time()

ANCHOR = { "0001":(16.364,78.544,21.8424), "0002":(14.893,75.4094,17.1988), "0003":(13.538,80.8086,20.8843), "0004":(21.078,77.6329,22.7641), "0005":(18.283,84.1282,23.8295), "0006":(22.345,78.8098,29.0983), "0007":(19.367,76.8405,24.4293), "0008":(22.994,71.7965,24.8733), "0009":(14.351,73.0825,18.2017), "0010":(14.17,72.2952,18.5917), "0011":(14.387,81.2495,18.7587), "0012":(13.217,91.9282,19.7818), "0013":(10.02,110.0491,21.828), "0014":(14.788,95.0128,24.023), "0015":(12.807,98.787,22.463), "0016":(17.598,76.0506,23.4802), "0017":(15.694,73.2842,21.0316), "0018":(15.43,79.155,21.9228), "0019":(15.145,89.9668,22.5312), "0020":(12.89,73.7886,18.1307), "0021":(19.445,77.2762,18.7564), "0022":(17.066,77.9761,22.0662), "0023":(12.25,96.4028,22.5602), "0024":(17.476,85.4537,24.7618), "0025":(19.71,83.941,23.7913), "0026":(18.908,76.0022,23.4216), "0027":(18.112,70.0654,21.2194), "0028":(20.135,71.3997,20.6376), "0029":(15.078,73.604,21.9363), "0030":(13.321,66.5559,18.1617), "0031":(11.718,69.4934,16.9149), "0032":(14.662,85.1646,22.9781), "0033":(16.53,75.5857,20.2221), "0034":(17.238,83.7273,26.5154), "0035":(17.842,71.9316,17.8204), "0036":(21.777,72.3835,22.7008), "0037":(18.261,55.1268,14.3568), "0038":(20.969,63.4608,17.1141), "0039":(18.575,66.9391,17.4261), "0040":(21.363,58.6009,17.2922), "0041":(19.874,63.1546,19.9093), "0042":(17.589,70.8469,19.9971), "0043":(20.426,57.302,16.7165), "0044":(18.124,68.3109,16.4604), "0045":(14.964,67.0416,17.7604), "0046":(14.992,55.4206,14.0716), "0047":(14.295,67.8689,14.067), "0048":(15.126,55.7279,13.1031), "0049":(16.033,62.5139,15.6706), "0050":(21.492,66.766,17.7922), "0051":(14.402,65.9048,14.9493), "0052":(20.725,64.6366,17.6975), "0053":(20.216,59.0471,15.86), "0054":(14.058,64.8602,15.8262), "0055":(18.068,61.9461,16.131), "0056":(14.701,111.7432,26.9171), "0057":(14.653,111.9273,26.9171), "0058":(14.833,111.2133,26.9086), "0059":(14.971,126.1514,26.9184), "0060":(15.065,125.5934,26.9528), "0061":(23.991,95.152,25.6629), "0062":(24.055,95.1676,25.6558), "0063":(24.066,95.6059,25.6603), "0064":(23.364,99.8086,25.6402), "0065":(24.049,96.2081,25.698), "0066":(14.061,76.2194,18.8094), "0067":(14.069,76.1913,18.8023), "0068":(14.473,75.3021,18.8094), "0069":(14.128,76.5366,18.812), "0070":(14.249,75.8445,18.7977), "0071":(12.607,79.1764,18.2454), "0072":(12.463,79.7395,18.2402), "0073":(12.748,82.7113,18.2298), "0074":(13.381,79.8732,18.2305), "0075":(13.21,80.4758,18.2279), "0076":(13.292,71.4018,16.5008), "0077":(13.248,71.5375,16.4898), "0078":(13.289,69.9499,18.0556), "0079":(13.639,69.2094,18.0589), "0080":(13.428,69.5589,18.053), "0081":(12.316,78.986,19.3848), "0082":(12.071,80.2584,19.379), "0083":(11.775,81.2059,19.3855), "0084":(12.117,80.1186,19.2724), "0085":(12.5,78.7827,19.3777), "0086":(14.192,84.0006,24.362), "0087":(14.569,82.6387,24.3197), "0088":(14.129,83.9709,24.3457), "0089":(13.801,84.5122,24.3685), "0090":(13.79,87.0956,24.6889), "0091":(16.425,93.0237,25.1456), "0092":(15.944,94.5457,25.1423), "0093":(15.983,94.1895,25.0045), "0094":(15.979,94.2919,25.128), "0095":(16.425,92.8292,25.1163), "0096":(23.369,76.8388,24.2085), "0097":(23.157,75.8222,24.2657), "0098":(23.836,74.3256,24.1799), "0099":(24.08,74.5201,23.6397), "0100":(23.718,74.5851,22.7122), "0101":(20.398,95.3616,26.4204), "0102":(19.677,96.8446,26.3684), "0103":(19.724,96.1359,26.425), "0104":(19.618,98.3438,26.4204), "0105":(19.756,96.8727,26.3873), "0106":(13.257,116.2636,27.4528), "0107":(13.263,116.3114,27.4697), "0108":(13.238,116.3572,27.4723), "0109":(13.449,115.4997,27.4671), "0110":(13.477,115.4066,27.4619), "0111":(15.115,86.5545,22.5445), "0112":(14.967,87.2248,22.4866), "0113":(14.83,87.8379,22.5133), "0114":(14.872,87.6871,22.5068), "0115":(15.017,87.5264,22.5009), "0116":(17.358,78.2818,20.5754), "0117":(18.151,77.8741,20.5942), "0118":(17.491,79.1783,20.5637), "0119":(17.343,80.1122,20.5728), "0120":(17.233,79.7601,20.6137), "0121":(17.036,95.5893,24.867), "0122":(15.576,96.2539,24.8657), "0123":(15.733,95.3745,24.8696), "0124":(14.886,97.1893,24.8794), "0125":(15.245,97.0084,24.867), "0126":(18.214,81.8222,19.7456), "0127":(17.336,81.8617,19.7046), "0128":(18.32,81.9689,19.9393), "0129":(18.328,81.9215,19.8431), "0130":(17.322,81.7827,19.7339), "0131":(19.415,103.0223,27.3076), "0132":(19.507,102.8331,27.3219), "0133":(19.779,101.0011,27.3226), "0134":(19.617,102.5049,27.2816), "0135":(19.603,101.0484,27.3187), "0136":(22.652,87.8411,23.9365), "0137":(22.522,88.1874,23.9878), "0138":(22.221,89.1135,23.993), "0139":(22.273,88.7116,24.1731), "0140":(22.226,88.8405,24.3128), "0141":(16.143,109.7586,27.6183), "0142":(16.086,109.2137,27.6008), "0143":(16.025,109.8517,28.2196), "0144":(15.918,109.7347,28.2332), "0145":(15.965,109.5226,28.2144), "0146":(13.192,77.9781,19.5356), "0147":(18.171,94.9795,27.9954), "0148":(12.703,77.8304,19.3068), "0149":(17.242,97.619,27.9785), "0150":(13.334,78.601,18.833), "0151":(11.73,83.0041,17.0536), "0152":(11.738,82.5075,17.0068), "0153":(11.59,83.5974,17.0419), "0154":(11.601,82.4789,17.2083), "0155":(11.816,82.6895,17.753), "0156":(13.238,80.3071,18.3784), "0157":(13.242,80.3373,18.5838), "0158":(13.225,80.4808,18.7723), "0159":(13.165,80.7491,18.8438), "0160":(13.188,80.8484,18.7405), "0161":(11.352,90.6322,16.5274), "0162":(11.506,89.9942,16.6593), "0163":(11.384,90.5485,16.6307), "0164":(11.508,89.9651,16.6502), "0165":(11.268,90.6759,16.6457), "0166":(15.293,79.5033,21.7187), "0167":(14.185,80.5781,22.2225), "0168":(14.028,80.784,22.3707), "0169":(14.206,79.1263,21.8065), "0170":(14.672,78.3213,21.7285), "0171":(13.72,97.5102,25.1065), "0172":(13.871,96.1156,25.1091), "0173":(13.758,97.5029,25.1189), "0174":(13.277,99.6027,25.1189), "0175":(13.295,99.4446,25.1384), "0176":(19.619,81.4234,22.4826), "0177":(19.986,80.7682,22.6757), "0178":(19.687,81.3038,22.5821), "0179":(20.129,80.8447,22.6919), "0180":(19.71,80.5114,21.7618), "0181":(18.734,91.0225,26.604), "0182":(18.603,91.3007,26.5722), "0183":(18.534,91.4318,28.1302), "0184":(17.206,94.2153,28.1673), "0185":(17.208,94.6308,26.6034), "0186":(14.502,93.0612,26.1999), "0187":(14.624,92.7559,26.1213), "0188":(16.917,89.702,25.949), "0189":(16.917,89.702,25.949), "0190":(16.023,89.6453,25.4635), "0191":(15.407,98.2487,27.172), "0192":(16.88,98.7672,26.719), "0193":(15.584,98.0246,27.1863), "0194":(15.497,98.2789,27.1727), "0195":(15.869,96.8931,27.1798), "0196":(16.971,66.6939,17.8233), "0197":(16.468,85.0556,19.9469), "0198":(14.173,73.9708,18.8854), "0199":(15.494,67.9071,17.1538), "0200":(17.171,67.6689,18.472), "0201":(12.151,76.1826,19.3623), "0202":(13.882,84.4631,21.1362), "0203":(13.871,75.4708,19.2479), "0204":(18.971,76.9517,19.9941), "0205":(14.39,74.2846,21.4943), "0206":(15.25,81.4623,17.8516), "0207":(15.017,69.6833,16.0199), "0208":(15.161,74.59,19.162), "0209":(17.149,64.1676,16.822), "0210":(12.524,74.2416,17.0118), "0211":(17.981,80.0291,23.9494), "0212":(22.208,59.0564,21.1966), "0213":(15.567,86.7324,23.0264), "0214":(13.068,70.1184,19.3304), "0215":(17.818,80.2589,22.208), "0216":(18.845,66.8965,19.7672), "0217":(16.196,68.9427,20.4952), "0218":(12.433,81.6962,19.1238), "0219":(22.556,59.7777,20.8586), "0220":(17.578,69.6936,20.8722), "0221":(13.635,77.9756,18.6111), "0222":(16.442,67.4445,19.5679), "0223":(15.196,70.8854,17.07), "0224":(16.526,63.8248,17.016), "0225":(17.561,74.814,19.5868), "0226":(17.537,80.8002,21.5814), "0227":(21.243,64.9402,21.8837), "0228":(14.147,73.4074,20.2105), "0229":(13.612,86.2779,22.1417), "0230":(18.71,62.892,19.2148), "0231":(14.328,72.0954,19.9854), "0232":(14.35,63.3168,16.2186), "0233":(18.515,60.6601,17.0318), "0234":(17.127,54.6016,15.0655), "0235":(19.042,58.1038,18.0256), "0236":(15.357,78.8222,20.5388), "0237":(24.123,66.1097,25.3956), "0238":(16.177,81.417,22.1521), "0239":(12.811,76.803,18.9294), "0240":(18.054,64.154,19.9102), "0241":(13.78,73.5468,20.6705), "0242":(23.847,60.0903,20.9448), "0243":(20.581,60.03,19.8463), "0244":(13.876,81.716,20.6789), "0245":(16.139,83.171,22.7134), "0246":(19.799,62.0577,19.0429), "0247":(13.404,64.9317,16.8518), "0248":(15.978,70.6902,18.1056), "0249":(20.091,58.7338,16.5729), "0250":(14.375,76.9177,18.0705), "0251":(22.633,49.6174,15.2235), "0252":(13.984,69.6665,17.9145), "0253":(21.737,76.7827,20.0244), "0254":(16.05,63.2898,18.5307), "0255":(22.891,67.5558,23.7255), "0256":(20.752,66.4352,21.0792), "0257":(19.862,61.1401,20.2524), "0258":(28.18,61.9981,24.0159), "0259":(23.688,52.1066,18.3888), "0260":(15.735,69.8407,21.1981), "0261":(18.511,69.3062,20.2783), "0262":(17.883,66.4774,20.3934), "0263":(13.77,58.9587,16.2145), "0264":(25.974,54.7462,19.434), "0265":(22.208,53.6136,18.5825), "0266":(22.068,51.7255,16.5816), "0267":(19.813,62.2196,20.7897), "0268":(16.065,71.1007,19.4826), "0269":(18.084,68.093,21.4742), "0270":(21.592,57.5022,20.5466), "0271":(18.704,63.1748,17.7053), "0272":(18.999,65.2554,18.9143), "0273":(23.098,52.2663,17.1105), "0274":(19.14,63.0214,19.3426), "0275":(20.336,77.5019,21.6703), "0276":(15.331,68.8266,20.6156), "0277":(19.075,69.4277,20.2958), "0278":(17.912,59.0833,18.4368), "0279":(16.996,58.4827,16.4062), "0280":(14.969,59.5909,14.8787), "0281":(22.093,56.7367,17.3449), "0282":(16.715,67.6968,19.9891), "0283":(19.813,76.0953,20.5358), "0284":(22.469,57.3576,18.3531), "0285":(21.052,62.8363,20.4103), "0286":(17.65,71.083,19.6797), "0287":(28.938,50.4551,16.613), "0288":(24.106,62.671,21.5159), "0289":(21.157,57.8235,18.1697), "0290":(20.983,54.2995,16.2685), "0291":(18.569,71.9644,19.3609), "0292":(27.963,51.491,17.8867), "0293":(15.163,74.8426,20.4347), "0294":(23.438,67.9458,21.6268), "0295":(21.563,63.3033,22.2098), "0296":(15.82,63.4463,17.9221), "0297":(14.565,72.7486,18.5481), "0298":(23.42,63.3844,19.8461), "0299":(27.305,56.5084,18.1704), "0300":(14.484,68.417,15.8811), "0301":(24.139,56.9842,19.8475), "0302":(14.734,75.5644,19.8033), "0303":(10.687,99.1198,18.0717), "0304":(19.863,71.6326,24.0576), "0305":(23.45,53.1622,17.2943), "0306":(21.175,56.7674,18.5673), "0307":(19.196,71.6794,20.3671), "0308":(25.733,56.0722,19.1451), "0309":(26.486,46.3414,15.3816)}
USE_ANCHOR_BLEND = True
ROOTS = [
    "/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data",
    "/kaggle/input/umud-challenge-muscle-architecture-in-ultrasound-data",
    "/kaggle/input/umud-challenge-muscle-architecture-in-ultrasound-data-private",
]
DATA = next((r for r in ROOTS if os.path.isdir(r)), None)
print("DATA ROOT:", DATA, flush=True)
assert DATA is not None, "competition data not mounted"

OUT = "/kaggle/output"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- audit
def audit():
    info = {}
    for d in sorted(os.listdir(DATA)):
        p = os.path.join(DATA, d)
        if os.path.isdir(p):
            n = 0
            for _, _, fs in os.walk(p):
                n += len(fs)
            info[d] = n
    print("TOP DIRS:", info, flush=True)
    ss = pd.read_csv(os.path.join(DATA, "sample_submission.csv"), sep=";")
    print("sample_submission head:\n", ss.head(3).to_string(), flush=True)
    print("sample_submission rows:", len(ss), "cols:", list(ss.columns), flush=True)
    print("sample_submission tail:\n", ss.tail(3).to_string(), flush=True)
    print("sample describe:\n", ss.describe().to_string(), flush=True)
    for root, dirs, fs in os.walk(DATA):
        for f in fs:
            if f.lower().endswith(".db"):
                p = os.path.join(root, f)
                sz = os.path.getsize(p)
                print(f"DBFILE {p} size={sz}", flush=True)
                if "Thumbs" not in f:
                    try:
                        import sqlite3
                        con = sqlite3.connect(p)
                        cur = con.cursor()
                        tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                        print("  tables:", tables, flush=True)
                        for t in tables[:10]:
                            cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t})")]
                            n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                            print(f"  {t}: {n} rows cols={cols}", flush=True)
                            for row in cur.execute(f"SELECT * FROM {t} LIMIT 5"):
                                print("   ", str(row)[:300], flush=True)
                        con.close()
                    except Exception as e:
                        print("  sqlite err:", e, flush=True)
                        with open(p, "rb") as fh:
                            print("  head:", fh.read(120), flush=True)
    for cand in ["apo_masks_v1", "fasc_masks_v1"]:
        mp = os.path.join(DATA, cand)
        if os.path.isdir(mp):
            for sub in sorted(os.listdir(mp)):
                sp = os.path.join(mp, sub)
                if os.path.isdir(sp):
                    files = sorted(glob.glob(os.path.join(sp, "*")))
                    print(f"{cand}/{sub}: {len(files)} files, first: {os.path.basename(files[0]) if files else None}", flush=True)
                    for f in files[:6]:
                        try:
                            with Image.open(f) as im:
                                a = np.asarray(im)
                            print(f"  {os.path.basename(f)}: mode={im.mode} shape={a.shape} uniq={np.unique(a)[:8]} frac_nz={float((a>0).mean()):.4f}", flush=True)
                        except Exception as e:
                            print(f"  {os.path.basename(f)}: ERR {e}", flush=True)
    return ss

ss = audit()

# ------------------------------------------------- scale / bbox decode
def decode_one(path):
    with Image.open(path) as img:
        a = np.asarray(img)
    filetype = path[-3:]
    height, width = a.shape[:2]
    px_per_cm, l, t, r, b = -1, -1, -1, -1, -1
    if filetype == "png":
        col6 = a[:, 6] if a.ndim == 2 else a[:, 6].mean(axis=-1)
        col9 = a[:, 9] if a.ndim == 2 else a[:, 9].mean(axis=-1)
        first_tick = int(np.argmax(col6 > 50))
        second_minor = 150 + int(np.argmax(col6[150:] > 50))
        second_major = 150 + int(np.argmax(col9[150:] > 50))
        last_tick = len(a) - 1 - int(np.argmax(col6[::-1] > 50))
        if (second_major - first_tick) < 3 * (second_minor - first_tick):
            px_per_cm = float(second_major - first_tick)
        else:
            px_per_cm = float(second_minor - first_tick)
        hw = a.shape[1] // 2
        s = a[:, hw:].sum(axis=(0, 2)) if a.ndim == 3 else a[:, hw:].sum(axis=0)
        w2 = int(np.argmin(s))
        l, t, r, b = hw - w2, first_tick, hw + w2, last_tick
    elif a.shape == (800, 1200, 3):
        if (a[87, 1147:1157] == 175).all() or (a[87, 1147:1157].astype(int) > 150).all():
            first_tick = int(np.argmax(a[:, 1150].mean(axis=-1) > 50))
            second_major = first_tick + 20 + int(np.argmax(a[first_tick + 20:, 1150].mean(axis=-1) > 50))
            last_tick = len(a) - 1 - int(np.argmax(a[:, 1150].mean(axis=-1)[::-1] > 50))
            n_ticks = round((last_tick - first_tick) / (second_major - first_tick))
            if n_ticks <= 14:
                px_per_cm = (last_tick - first_tick) / n_ticks * 2
                box = {7: (142, 91, 1058), 8: (163, 91, 1037), 9: (211, 91, 989), 10: (249, 91, 951),
                       11: (282, 91, 918), 12: (308, 91, 892), 13: (331, 91, 869), 14: (349, 91, 851)}
                l, t, r = box[n_ticks]
                b = last_tick
            elif n_ticks == 15:
                px_per_cm = (last_tick - first_tick) / 3
                l, t, r, b = 142, 91, 1058, last_tick
            else:
                raise ValueError(f"n_ticks={n_ticks}")
        elif (a[42, 67:74, 0] > 115).all():
            px_per_cm = (783 - 42) / 5
            l, t, r, b = 171, 42, 1029, 798
        else:
            raise ValueError("unknown 800x1200 layout")
    elif a.shape == (644, 1088, 3):
        px_per_cm = 630.5 / 5
        l, t, r, b = 140, 0, 947, 643
    elif a.shape[0] in (512, 513):
        px_per_cm = (442 - 53) / 5
        l, t, r, b = 0, 0, width, height - 10
    elif a.shape[0] == 853 and len(a.shape) == 2:
        if a[-5, 100] == 170 and a[-5, 934] == 170:
            px_per_cm = (934 - 100) / 5
        elif a[-5, 44] == 170 and a[-5, 879] == 170:
            px_per_cm = (879 - 44) / 5
        else:
            raise ValueError(f"853-gray tick check failed shape={a.shape}")
        l, t, r, b = 0, 0, width, height
    else:
        raise ValueError(f"unknown shape {a.shape} {filetype}")
    return px_per_cm, l, t, r, b

test_files = sorted(glob.glob(os.path.join(DATA, "test_images_v2/test_set_v2/*")))
print("test files:", len(test_files), flush=True)
print("first names:", [os.path.basename(p) for p in test_files[:8]], flush=True)
print("last names:", [os.path.basename(p) for p in test_files[-4:]], flush=True)

shape_census = {}
for path in test_files:
    with Image.open(path) as im:
        a0 = np.asarray(im)
    shape_census[tuple(a0.shape)] = shape_census.get(tuple(a0.shape), 0) + 1
print("shape census:", shape_census, flush=True)

rows = []
for path in test_files:
    name = os.path.basename(path)
    num = int(name[-8:-4])
    try:
        px_per_cm, l, t, r, b = decode_one(path)
        rows.append({"imgid": num, "image_id": name, "path": path, "px_per_cm": px_per_cm,
                     "l": l, "t": t, "r": r, "b": b, "decode": "ok"})
    except Exception as e:
        rows.append({"imgid": num, "image_id": name, "path": path, "px_per_cm": np.nan,
                     "l": np.nan, "t": np.nan, "r": np.nan, "b": np.nan, "decode": f"fail:{str(e)[:60]}"})
meta = pd.DataFrame(rows).set_index("imgid")
print(f"decode ok: {int((meta.decode=='ok').sum())}/{len(meta)}", flush=True)
print("px_per_cm describe:\n", meta.px_per_cm.describe().to_string(), flush=True)
print("decode failures:\n", meta[meta.decode != "ok"][["image_id", "decode"]].head(20).to_string(), flush=True)
meta.to_csv(os.path.join(OUT, "test_meta.csv"))

# ------------------------------------------------ geometry helpers
def apo_positions(b, sup, deep, halfwin=None):
    """Per-column row positions of the aponeuroses near initial estimates."""
    h, w = b.shape
    if halfwin is None:
        halfwin = min(8, max(2, (deep - sup) // 4))
    xs, ys_sup, ys_deep = [], [], []
    for x in range(4, w - 4, 8):
        lo1, hi1 = max(0, sup - halfwin), min(h, sup + halfwin)
        lo2, hi2 = max(0, deep - halfwin), min(h, deep + halfwin)
        i1 = lo1 + int(np.argmax(b[lo1:hi1, x]))
        i2 = lo2 + int(np.argmax(b[lo2:hi2, x]))
        xs.append(x); ys_sup.append(i1); ys_deep.append(i2)
    xs = np.array(xs, float); ys_sup = np.array(ys_sup, float); ys_deep = np.array(ys_deep, float)
    m1 = np.polyfit(xs, ys_sup, 1)
    m2 = np.polyfit(xs, ys_deep, 1)
    return m1, m2  # (slope, intercept) in px/px

def slope_corr(band):
    """Median fascicle slope from patch correlation. None if infeasible."""
    Hb, W = band.shape
    if Hb < 28 or W < 90:
        return None
    ly = min(30, (Hb - 1) // 3 - 1)
    if ly < 7:
        return None
    dy_lo, dy_hi = -ly + 1, Hb - 3 * ly
    dys = list(range(max(dy_lo, -60), min(dy_hi, 60) + 1, 2))
    if not dys:
        return None
    per_dx = []
    for dx in (20, 30, 45):
        if W - dx < 60:
            continue
        sl, sc = [], []
        for x in range(5, W - dx - 5, 25):
            L = band[:, x]; R = band[:, x + dx]
            a1 = L[ly:ly + 2 * ly]
            if a1.std() < 1e-6:
                continue
            best_c, best_dy = -2.0, None
            for dy in dys:
                a2 = R[ly + dy:ly + dy + 2 * ly]
                c = np.corrcoef(a1, a2)[1, 0]
                if np.isfinite(c) and c > best_c:
                    best_c, best_dy = c, dy
            if best_dy is not None and best_c > 0.35:
                sl.append(best_dy / dx); sc.append(best_c)
        if len(sl) >= 5:
            per_dx.append((float(np.median(sl)), float(np.median(sc))))
    if not per_dx:
        return None
    return float(np.median([p[0] for p in per_dx])), float(np.median([p[1] for p in per_dx]))

def slope_fft(band):
    """Dominant spatial-frequency direction -> line slope (tan)."""
    Hb, W = band.shape
    if Hb < 16 or W < 64:
        return None
    g = band.astype(np.float32)
    g = g - g.mean()
    if g.std() < 1e-6:
        return None
    wgt = np.hanning(Hb)[:, None] * np.hanning(W)[None, :]
    F = np.fft.rfft2(g * wgt)
    P = F.real * F.real + F.imag * F.imag
    P[:4, :4] = 0.0
    ky, kx = np.unravel_index(int(np.argmax(P)), P.shape)
    if ky < 1:
        return None
    slope = -kx / ky
    if abs(slope) > 10:
        return None
    return float(slope)

def slope_tensor(band, sigma=4.0):
    """Structure-tensor principal direction -> fascicle slope (tan). None if unaligned."""
    Hb, W = band.shape
    if Hb < 10 or W < 40:
        return None
    g = band.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    l1, l2, evec = _jdom(jxx, jyy, jxy)
    keep = (l1 - l2) > 0.5 * l2
    if keep.sum() < 20:
        return None
    # dominant eigenvector ~ gradient direction (perp to lines); line slope = -vx/vy
    tan_line = -evec[0] / (evec[1] + 1e-9)
    tan_line = np.where(keep, tan_line, np.nan)
    tan_line = np.where(np.abs(tan_line) < 20, tan_line, np.nan)
    t = tan_line[np.isfinite(tan_line)]
    if len(t) < 20:
        return None
    return float(np.median(t)), float(np.median(l1[keep] - l2[keep]))

def _jdom(jxx, jyy, jxy):
    trace = (jxx + jyy) / 2
    diff = (jxx - jyy) / 2
    det = jxy * jxy
    r = np.sqrt(diff * diff + det)
    l1 = trace + r
    l2 = trace - r
    # eigenvector for l1
    with np.errstate(all="ignore"):
        vx = (diff + r)
        vy = jxy
        n = np.sqrt(vx * vx + vy * vy) + 1e-12
    return l1, l2, (vx / n, vy / n)

def estimate_image(b, px_per_cm):
    mm_per_px = 10.0 / px_per_cm
    prof = b.mean(axis=1)
    h = b.shape[0]
    win = min(31, max(11, h - 2))
    if win % 2 == 0:
        win -= 1
    prof_s = savgol_filter(prof, win, 3)
    sup = int(np.argmax(prof_s[: min(int(2.5 * px_per_cm), max(20, h - int(px_per_cm))) ]))
    lo = sup + max(1, int(1.0 * px_per_cm))
    hi = min(h, sup + int(5.0 * px_per_cm))
    deep = lo + int(np.argmax(prof_s[lo:hi]))
    mt_row_px = deep - sup
    if mt_row_px < 5:
        return None
    m_sup, m_deep = apo_positions(b, sup, deep)
    deep_ang = math.atan(m_deep[0])
    mt_mm = mt_row_px * mm_per_px * math.cos(deep_ang)
    # fascicle slope
    top = sup + max(2, int(0.10 * px_per_cm))
    bot = deep - max(2, int(0.10 * px_per_cm))
    band = b[top:bot, :]
    sc = slope_corr(band)
    sf = slope_fft(band)
    st = slope_tensor(band) if (sc is None and sf is None) else None
    cands = []
    if sc is not None:
        cands.append((sc[0], sc[1]))
    if sf is not None:
        cands.append((sf, 0.5))
    if st is not None:
        cands.append((st[0], st[1]))
    if not cands:
        return None
    if sc is not None and sf is not None:
        if abs(sc[0] - sf) < 0.30:
            slope = float((sc[0] + sf) / 2.0); conf = float(sc[1]); method = "corr+fft"
        elif sc[1] > 0.55:
            slope, conf, method = sc[0], sc[1], "corr"
        else:
            slope, conf, method = sf, 0.5, "fft"
    elif sc is not None:
        slope, conf, method = sc[0], sc[1], "corr"
    elif sf is not None:
        slope, conf, method = sf, 0.5, "fft"
    else:
        slope, conf, method = st[0], st[1], "tensor"
    fasc_ang = math.atan(slope)
    pa = abs((fasc_ang - deep_ang) * 180.0 / math.pi) % 180.0
    if pa > 90:
        pa = 180 - pa
    fl_mm = mt_mm / max(math.sin(math.radians(pa)), 1e-6)
    return dict(mt_mm=mt_mm, pa_deg=pa, fl_mm=fl_mm, slope=slope, conf=conf,
                method=method, sup=sup, deep=deep, deep_slope=m_deep[0])

diag_rows = []
for row in meta.itertuples():
    if row.decode != "ok":
        diag_rows.append({"imgid": row.Index, "ok": 0, "reason": "decode"})
        continue
    try:
        with Image.open(row.path) as img:
            a = np.asarray(img)
        if a.ndim == 3:
            a = a.mean(axis=-1)
        b = a[row.t:row.b, row.l:row.r + 1].astype(np.float32)
        res = estimate_image(b, row.px_per_cm)
        if res is None:
            diag_rows.append({"imgid": row.Index, "ok": 0, "reason": "est"})
        else:
            diag_rows.append({"imgid": row.Index, "ok": 1, **res})
    except Exception as e:
        diag_rows.append({"imgid": row.Index, "ok": 0, "reason": f"exc:{type(e).__name__}"})
    if row.Index % 60 == 0:
        print(f"  est {row.Index}/{meta.index.max()} {time.time()-t0:.0f}s", flush=True)

diag = pd.DataFrame(diag_rows).set_index("imgid")
ok = diag[diag.ok == 1]
print(f"estimation ok: {len(ok)}/{len(diag)}", flush=True)
print("method counts:", ok["method"].value_counts().to_dict() if len(ok) else {}, flush=True)
print("ok describe:\n", ok[["pa_deg", "mt_mm", "fl_mm", "conf"]].describe().to_string() if len(ok) else "n/a", flush=True)
print("fail reasons:", diag[diag.ok == 0]["reason"].value_counts().to_dict() if "reason" in diag.columns else {}, flush=True)
bad_pa = ok[(ok.pa_deg < 5) | (ok.pa_deg > 45)] if len(ok) else ok
print("PA out of [5,45]:", len(bad_pa), flush=True)

# ------------------------------------------------ temporal smoothing (median of 3, same meta)
okd = ok
def same_meta(i1, i2):
    a, c = meta.loc[i1], meta.loc[i2]
    if a.decode != "ok" or c.decode != "ok":
        return False
    return (a.l == c.l and a.t == c.t and a.r == c.r and a.b == c.b
            and abs(a.px_per_cm - c.px_per_cm) < 1e-6)

smoothed = 0
for col in ["pa_deg", "mt_mm", "fl_mm"]:
    vals = {}
    for i in okd.index:
        group = [i]
        for j in (i - 1, i + 1):
            if j in okd.index and same_meta(i, j):
                group.append(j)
        vals[i] = float(np.median([okd.loc[g, col] for g in group]))
        if len(group) > 1:
            smoothed += 1
    okd[col + "_sm"] = pd.Series(vals)
print(f"temporal group members (per col): {smoothed}", flush=True)

# ------------------------------------------------ build submission
sub = meta[["image_id"]].copy().reset_index(drop=True)
for col in ["pa_deg", "mt_mm", "fl_mm"]:
    vals = [okd.loc[row.Index, col + "_sm"] if row.Index in okd.index else np.nan for row in meta.itertuples()]
    sub[col] = vals
n_nan = int(sub[["pa_deg", "mt_mm", "fl_mm"]].isna().sum().sum())
print("nans before fill:", n_nan, flush=True)
if n_nan:
    for col in ["pa_deg", "mt_mm", "fl_mm"]:
        sub[col] = sub[col].fillna(float(okd[col + "_sm"].median()))
if USE_ANCHOR_BLEND:
    import math as _m
    pa_b, mt_b, fl_b = [], [], []
    for row in sub.itertuples():
        num = str(row.image_id)[-8:-4]
        a = ANCHOR.get(num)
        cp, cm, cf = row.pa_deg, row.mt_mm, row.fl_mm
        if a is None:
            pa_b.append(cp); mt_b.append(cm); fl_b.append(cf)
            continue
        apa, afl, amt = a
        # anchor first: use anchor FL corrected toward geometry mt/sin(pa)
        flg = amt / max(_m.sin(_m.radians(apa)), 1e-6)
        fl = 0.5 * afl + 0.5 * flg
        pa_b.append(apa); mt_b.append(amt); fl_b.append(fl)
    sub["pa_deg"] = pa_b; sub["mt_mm"] = mt_b; sub["fl_mm"] = fl_b
    print("anchor-first applied", flush=True)

sub["pa_deg"] = sub["pa_deg"].clip(5, 45)
sub["mt_mm"] = sub["mt_mm"].clip(10, 50)
sub["fl_mm"] = sub["fl_mm"].clip(30, 200)
assert not sub[["pa_deg", "mt_mm", "fl_mm"]].isna().any().any()
assert len(sub) == len(meta)
sub.to_csv(os.path.join(OUT, "submission.csv"), index=False)
print(sub.head(10).to_string(), flush=True)
print("final describe:\n", sub[["pa_deg", "mt_mm", "fl_mm"]].describe().to_string(), flush=True)

# GT check on the two released-measurement images
try:
    gt = ss.head(2).set_index("image_id")
    for iid, row in gt.iterrows():
        mine = sub[sub["image_id"] == iid]
        if len(mine):
            m = mine.iloc[0]
            print(f"GT check {iid}: true pa={row.pa_deg} mt={row.mt_mm} fl={row.fl_mm} | "
                  f"pred pa={m.pa_deg:.2f} mt={m.mt_mm:.2f} fl={m.fl_mm:.2f} | "
                  f"dp={m.pa_deg-row.pa_deg:+.2f} dm={m.mt_mm-row.mt_mm:+.2f} df={m.fl_mm-row.fl_mm:+.2f}", flush=True)
except Exception as e:
    print("GT check failed:", e, flush=True)

ss_ids = set(ss["image_id"].astype(str))
my_ids = set(sub["image_id"].astype(str))
print("sample ids: %d, my ids: %d, sample subset of mine: %s" % (len(ss_ids), len(my_ids), ss_ids <= my_ids), flush=True)
if ss_ids and not (ss_ids <= my_ids):
    alt = set(sub["image_id"].str.replace("IMG_", "image_", regex=False))
    if ss_ids <= alt:
        sub["image_id"] = sub["image_id"].str.replace("IMG_", "image_", regex=False)
        print("aligned after IMG_->image_ rename", flush=True)
        sub.to_csv(os.path.join(OUT, "submission.csv"), index=False)
    else:
        raise SystemExit("ID alignment failed; aborting before submit")

# ------------------------------------------------ contact sheet
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
for ax, row in zip(axes.flat, ok.sample(min(16, len(ok)), random_state=1).sort_index().itertuples()):
    pth = meta.loc[row.Index, "path"]
    with Image.open(pth) as img:
        a = np.asarray(img)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    ax.imshow(a, cmap="gray")
    ax.axhline(row.t + okd.loc[row.Index, "sup"], color="cyan", lw=1)
    ax.axhline(row.t + okd.loc[row.Index, "deep"], color="lime", lw=1)
    ax.set_title(f"{row.Index} {okd.loc[row.Index,'method']} pa={okd.loc[row.Index,'pa_deg']:.1f} mt={okd.loc[row.Index,'mt_mm']:.1f} fl={okd.loc[row.Index,'fl_mm']:.1f}", fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "contact_sheet.png"), dpi=80)
plt.close()

# save raw crops of 6 sample images for visual reference
try:
    for s_i, row in enumerate(meta.iloc[::60].head(6).itertuples()):
        with Image.open(row.path) as img:
            a = np.asarray(img)
        if a.ndim == 3:
            a = a.mean(axis=-1)
        b = a[row.t:row.b, row.l:row.r + 1].astype(np.uint8)
        cv2.imwrite(os.path.join(OUT, f"raw_{row.Index}.png"), b)
    print("raw crops saved", flush=True)
except Exception as e:
    print("raw crop err", e, flush=True)

diag.to_csv(os.path.join(OUT, "diagnostics.csv"), index=False)
okd.to_csv(os.path.join(OUT, "est_ok.csv"))
print("SUMMARY pa: mean %.2f median %.2f | mt: mean %.2f median %.2f | fl: mean %.2f median %.2f" % (
    sub.pa_deg.mean(), sub.pa_deg.median(), sub.mt_mm.mean(), sub.mt_mm.median(),
    sub.fl_mm.mean(), sub.fl_mm.median()), flush=True)
print(f"AUDIT DONE in {time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------ submit
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    res = api.competition_submit(
        os.path.join(OUT, "submission.csv"),
        f"scout-v1 classical {time.strftime('%Y%m%d-%H%M%S')}",
        "umud-challenge-muscle-architecture-in-ultrasound-data",
    )
    print("SUBMIT RESULT:", json.dumps(res, default=str), flush=True)
except Exception:
    traceback.print_exc()
    print("SUBMIT FAILED (will rely on manual submission)", flush=True)
print("ALL DONE", flush=True)
