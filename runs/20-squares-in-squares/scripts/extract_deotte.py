"""Extract Deotte's 543 packings from his multi-gadget-DP notebook cell 1."""
import base64
import builtins
import csv
import hashlib
import io
import json
import lzma
import math
import sys
import time
import types
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# stub IPython
ipy = types.ModuleType("IPython")
disp = types.ModuleType("IPython.display")
def _display(*a, **k): pass
def _html(*a, **k): return ""
disp.display = _display
disp.HTML = _html
ipy.display = disp
sys.modules["IPython"] = ipy
sys.modules["IPython.display"] = disp

# stub matplotlib
mp_ = types.ModuleType("matplotlib")
plt = types.ModuleType("matplotlib.pyplot")
plt.rcParams = types.SimpleNamespace(update=lambda *a, **k: None)
mp_patches = types.ModuleType("matplotlib.patches")
mp_collections = types.ModuleType("matplotlib.collections")
mp_collections.PatchCollection = object
mp_colors = types.ModuleType("matplotlib.colors")
class _LSCM:
    @classmethod
    def from_list(cls, *a, **k):
        return None
mp_colors.LinearSegmentedColormap = _LSCM
mp_.pyplot = plt
mp_.patches = mp_patches
mp_.collections = mp_collections
mp_.colors = mp_colors
sys.modules["matplotlib"] = mp_
sys.modules["matplotlib.pyplot"] = plt
sys.modules["matplotlib.patches"] = mp_patches
sys.modules["matplotlib.collections"] = mp_collections
sys.modules["matplotlib.colors"] = mp_colors

nb = json.load(open("logs/deotte_nb/squares-in-squares-multi-gadget-fault-dp-22-48.ipynb"))
src = "".join(nb["cells"][1]["source"])
g = {}
exec(compile(src, "cell1", "exec"), g)

bank = g["CHAMPION_BANK"]
L = g["CHAMPION_L"]
out = {}
for n, p in bank.items():
    out[str(n)] = [[float(x), float(y), float(a)] for x, y, a in p]
json.dump({"L": {str(i + 1): float(v) for i, v in enumerate(L)},
           "packings": out},
          open("models/deotte_bank.json", "w"))
print("bank size:", len(bank), "rows:", sum(len(v) for v in bank.values()))
print("champ total score:", float(sum(L[n - 1] / math.sqrt(n) - 1 for n in range(1, 1001))))
