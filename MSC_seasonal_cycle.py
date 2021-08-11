# -*- coding: utf-8 -*-
'''
Precipitation changes in high southern latitudes

This script reads CMIP6 (AMIP & Hist) and reanalysis datasets and
calculates their ensemble means.
Then, for the Antarctic regions defined in mscbib [Austral, EAIS, Ross,
Weddell, and WAIS], the monthly data is grouped by month in order to
address the annual cycle.
This script plots: (i) ensemble mean annual cycle line plot
                   (ii) bar plot annual cycle for each dataset

Note: always calculate reanalysis-ensemble mean and model-ensemble
mean separately. Do not mix different data types.

Natália Silva (2021)
natalia3.silva@usp.br
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mscbib import clipper, my_cmap
import glob
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# print(plt.style.available)
plt.style.use('seaborn-darkgrid')
plt.rcParams["figure.figsize"] = (14, 4)
sns.set_context('talk', font_scale=1)


# *_*_*_*_*_*_*_*_*_*_*   READ AND CONCATENATE DATA   _*_*_*_*_*_*_*_*_*_*
# *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_

dts = {'Rean': [], 'Hist': [], 'AMIP': []}

# ####### Reanalysis #######
# ##########################
files = sorted(glob.glob('/Volumes/snati/data/PPT/reanalise/monthly/pr_*.nc'))
dset = []
for f in files:
    ds = xr.open_dataset(f, drop_variables=[
                         'initial_time0_encoded', 'time_bnds'])
    dset.append(ds)


# Change calendars for monthly data
for i in range(len(dset)):
    dset[i] = dset[i].sel(lat=slice(-89.5, -40), time=slice('1979', '2014'))
    ti, tf = dset[i].time[0], dset[i].time[-1].values + np.timedelta64(31, 'D')
    dset[i]['time'] = np.arange(ti.values, tf, dtype='datetime64[M]')


rean = ['20thCV3', 'CFSR', 'ERA-20', 'ERA-I', 'JRA55', 'MERRA2', 'NCEP2']
dts['Rean'] = xr.concat(dset,
                        'realization').assign_coords({'realization': rean})
dts['Rean'] = xr.concat([dts['Rean'], dts['Rean'].mean('realization').assign_coords({'realization': 'Ens.M'})], dim='realization')


# ####### CMIP hist/amip #######
# ##############################
path = ['/Volumes/snati/data/PPT/cmip6_hist/amip/monthly/pr*.nc',
        '/Volumes/snati/data/PPT/cmip6_hist/hist/monthly/pr*.nc']

for p, n in zip(path, list(dts.keys())[1:]):
    files = sorted(glob.glob(p))
    dset, name = [], []
    for f in files:
        ds = xr.open_dataset(f)
        ds = ds.assign_coords({'realization': ds.source_id})
        ds = ds.sel(lat=slice(-89.5, -40.5), time=slice('1979', '2014'))
        ds['time'] = np.arange('1979-01', '2015-01', dtype='datetime64[M]')
        dset.append(ds)
        name.append(ds.source_id)
        #
        #
    dts[n] = xr.concat(dset, 'realization')
    dts[n] = xr.concat([dts[n], dts[n].mean('realization').assign_coords({'realization': 'Ens.M'})], dim='realization')
    #
    #


del(files, f, ds, dset, glob, rean)


# *_*_*_*_*_*_*_*_ MONTHLY ANALYSIS OF PRECIPITATION   _*_*_*_*_*_*_*_*_*_
# *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_


# Regional time series
loc = ['Austral', 'EAIS', 'W-WAIS_Ross', 'E-WAIS_Amundsen', 'Weddell Sea']
l = 4

reg = {'Rean': [], 'Hist': [], 'AMIP': []}
for i in dts:
    r, _, _ = clipper(dts[i].groupby('time.month').mean('time').pr, loc[l])
    reg[i] = r.mean(['lat', 'lon']).to_pandas()
    #


# *_*_*_*_*_*_*_*_ PLT  _*_*_*_*_*_*_*_*_*_
# *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_

mon = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
       'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
idx = reg['Rean']['Ens.M'].index
cor = ['R', 'm']
c3, ls = ['goldenrod', 'turquoise', 'coral'], ['-', ':', '--']

# for n in range(len(c3)):
#     plt.bar(idx * 4 - n, list(reg.values())[n]['Ens.M'], color=c3[n])
#     plt.plot(idx * 4 - 1, list(reg.values())[n]['Ens.M'], linewidth=2,
#              color='black', alpha=0.6, ls=ls[n], label=list(reg.keys())[n])
#     #

# plt.xticks(idx * 4 - 1, mon, rotation=45)
n = 1
c = my_cmap(25, cor[n])
ax = list(reg.values())[n].plot.bar(color=c, legend=None)
plt.plot(idx - 1, list(reg.values())[n]['Ens.M'], linewidth=2, color='black',
         linestyle='-', label=list(reg.keys())[n], alpha=0.6)
plt.plot(idx - 1, list(reg.values())[0]['Ens.M'], linewidth=2, color='black',
         linestyle='--', label=list(reg.keys())[0], alpha=0.6)
plt.xticks(idx - 1, mon, rotation=45)
# plt.ylim([0, 0.25])
plt.subplots_adjust(left=0.07, bottom=0.25, right=0.8)
plt.xlabel('months')
plt.ylabel('PPT [mm day$^-$$^1]$')
plt.title(loc[l])
plt.legend(bbox_to_anchor=(0.15, 0.1, 1.15, 1.1))
plt.show()
# plt.savefig('cycle' + loc[l] + '.pdf', dpi=50, facecolor='w', format='pdf')
plt.close()


# FIM
