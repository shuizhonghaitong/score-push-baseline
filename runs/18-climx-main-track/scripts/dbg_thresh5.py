import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
import netCDF4 as nc
import xarray as xr
from climx_common import lite_file
from climx_indices import base_period_slice, _hyndman8, percentile_doy_window

ds2 = nc.Dataset(str(lite_file("hist_late")))
base = base_period_slice({v: np.asarray(ds2.variables[v][0], dtype=np.float64) for v in ["tasmax"]})
ds2.close()
a = base["tasmax"]; T = a.shape[0]
doy_k, i, j = 364, 10, 10
half = 2
out=[]
for y in range(30):
    c = y*365 + doy_k
    lo2, hi2 = max(c-half,0), min(c+2, T-1)
    out.append(a[lo2:hi2+1, i, j])
vec = np.concatenate(out)
print("direct _hyndman8:", _hyndman8(vec, 0.90))
# now what apply_along_axis does
arr = np.stack([a[(max(y*365+doy_k-2,0)):(min(y*365+doy_k+2,T-1)+1)] for y in range(30)], axis=0)  # wrong concat? no: concatenate
arr = np.concatenate([a[(max(y*365+doy_k-2,0)):(min(y*365+doy_k+2,T-1)+1)] for y in range(30)], axis=0)
print("arr shape:", arr.shape)
res = np.apply_along_axis(lambda c: _hyndman8(c, 0.90), 0, arr)
print("apply_along_axis result shape:", res.shape)
print("res[i,j]:", res[i, j], " res[0,0]:", res[0,0])
# what is res actually? compare to per-cell direct
direct = np.array([[_hyndman8(arr[:, ii, jj], 0.90) for jj in range(18)] for ii in range(12)])
print("apply==direct?", np.allclose(res, direct), " maxdiff:", np.abs(res-direct).max())
# full percentile_doy_window at this cell
mine = percentile_doy_window(a, 0.90)
print("percentile_doy_window[364,10,10]:", mine[364, 10, 10])
print("xclim expected ~263.34")
