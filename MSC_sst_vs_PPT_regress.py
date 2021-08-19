# -*- coding: utf-8 -*-
'''
Precipitation changes in high southern latitudes

This script reads reanalyses and CMIP6 PPT and SST datasets as ensamble,
and calculates their ens. mean.
Then, for each defined Antarctic region [Austral, EAIS, Ross, Wed and WAIS],
it estimates the variables linear trend and secondary statistics.
At last, this script plots: PPT trend X SST trend, and their correlation
as R2.

Natália Silva (2021)
natalia3.silva@usp.br
'''

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from mscbib import my_cmap, ptext, norm_cmap, mod_agr, clipper  # linear_trend3
import glob
import seaborn as sns
from scipy import stats


# Figure setups
pd.set_option('display.float_format', lambda x: '%.2e' % x)
sns.set_context('talk', font_scale=1)  # large fontsize
plt.rcParams['hatch.linewidth'] = 0.3
plt.rcParams['hatch.color'] = 'gray'
plt.rcParams["figure.figsize"] = (10, 8)


# *_*_*_*_*_*_*_*_*_*_*   READ AND CONCATENATE DATA   _*_*_*_*_*_*_*_*_*_*
# *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_

# # ####### CMIP6 pr + tos #######
# path = ['/Volumes/snati/data/PPT/pmi4-cmip6/hist/pr*.nc',
#         '/Volumes/snati/data/SST/cmip6_hist/*.nc']
# m = {'pr': [], 'tos': []}

# for p, n in zip(path, m):
#     files = sorted(glob.glob(p))
#     dset = []
#     for f in files:
#         ds = xr.open_dataset(f, drop_variables=[
#                              'time_bnds', 'initial_time0_encoded'])
#         ds = ds.assign_coords({'realization': ds.source_id})
#         ds = ds.sel(lat=slice(-89.5, -40.5), time=slice('1900', '2014'))
#         ds['time'] = ds.time.dt.year
#         dset.append(ds)
#         #
#         #
#     m[n] = xr.concat(dset, 'realization')
#     m[n] = xr.concat([m[n], m[n].mean('realization').assign_coords(
#         {'realization': 'Ens.M'})], dim='realization')
#     #


# ####### Reanalysis #######
path = ['/Volumes/snati/data/PPT/reanalise/pr*.nc',
        '/Volumes/snati/data/SST/obs-rean/sst*.nc']
rname = ['20Cv3', 'CFSR', 'Era20C', 'EraInt', 'JRA55', 'MERRA2', 'NCEP2']
rean = {'pr': [], 'tos': []}

for p, n in zip(path, rean):
    files = sorted(glob.glob(p))
    dset = []
    for f in files:
        ds = xr.open_dataset(f, drop_variables=[
                             'time_bnds', 'initial_time0_encoded'])
        ds = ds.sel(lat=slice(-89.5, -40.5), time=slice('1900', '2014'))
        ds['time'] = ds.time.dt.year
        if n == 'tos':
            ds = ds.where(ds.tos > -10)
        dset.append(ds)
        #
        #
    rean[n] = xr.concat(dset, 'realization').assign_coords({'realization': rname})
    rean[n] = xr.concat([rean[n], rean[n].mean('realization').assign_coords(
        {'realization': 'REAN'})], dim='realization')


# # # ####### Observations #######
# path = ['/Volumes/snati/data/PPT/reanalise/O*.nc',
#         '/Volumes/snati/data/SST/obs-rean/O*.nc']
# obs = {'pr': [], 'tos': []}

# for p, n in zip(path, m):
#     files = sorted(glob.glob(p))
#     dset = []
#     for f in files:
#         ds = xr.open_dataset(f, drop_variables=[
#                              'time_bnds', 'initial_time0_encoded'])
#         ds = ds.sel(lat=slice(-89.5, -40.5), time=slice('1900', '2014'))
#         ds['time'] = ds.time.dt.year
#         dset.append(ds)
#         #
#         #
#     o = xr.concat(dset, 'realization')
#     obs[n] = o.mean('realization').assign_coords({'realization': 'Obs'})


# del(ds, dset, f, files, glob, obs, p, path)


# *_*_*_*_*_*_*_ CALCULATE TREND PER REGION PER DATA   _*_*_*_*_*_*_*_*_*_
# *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_
# # Regional time series
d = rean

name = d['pr'].realization.values
loc = {'pr': ['Austral', 'EAIS', 'W-WAIS_Ross', 'E-WAIS_Amundsen', 'Weddell Sea'],
       'tos': ['Austral', 'Austral', 'Austral', 'Austral', 'Austral']}
idx = ['slpB80', 'bB80', 'rB80', 'pB80', 'errB80', 'slpA80', 'bA80',
       'rA80', 'pA80', 'errA80']
l = 4
reg = {'pr': [], 'tos': []}
# Empty dset to be filled with statistics results for each dataset
res = {'pr': pd.DataFrame(columns=name, index=idx),
       'tos': pd.DataFrame(columns=name, index=idx)}

for i in d:
    r, _, _ = clipper(d[i][i], loc[i][l])
    # est[i] = ens.ensemble_mean_std_max_min(r.to_dataset())  # xr.Dataset
    reg[i] = r.mean(['lat', 'lon']).to_pandas().T
    #
    for column in reg[i]:
        qi = reg[i][column].loc[1850:1979].dropna()
        if len(qi) <= 1:
            res[i][column][0:5] = np.nan
        elif len(qi) > 1:
            res[i][column][0:5] = stats.linregress(qi.index, qi)
        qf = reg[i][column].loc[1980:].dropna()
        res[i][column][5:10] = stats.linregress(qf.index, qf)
        #
        #
    #
    res[i].loc['slpB80'] *= 10
    res[i].loc['slpA80'] *= 10
    #


# # *_*_*_*_*_*_*_  SST vs PPT trends   *_*_*_*_*_*_
# Regression analysis between ppt and sst trends
# "How much of the ppt change can be explained by the sst change?"
c = my_cmap(25, 'r')
# plt.style.use('seaborn-darkgrid')
plt.style.use('seaborn-white')

per, tex = ['slpB80', 'slpA80'], ['BEF 80: ', 'AFT 80: ']
# # Austral
# yl2 = [-0.02, 0.07]
# xl2 = [-0.3, 0.05]
# EAIS
# yl2 = [-0.01, 0.01]
# # Ross
# yl2 = [-0.045, 0.055]
# # WAIS
# yl2 = [-0.07, 0.07]
# # Weddell
yl2 = [-0.01, 0.06]


for p, t in zip(per, tex):
    fig = plt.figure()
    x = res['tos'].loc[p].astype(float)
    y = res['pr'].loc[p].astype(float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    a, b, r, pv, _ = stats.linregress(x[mask], y[mask])
    fit = a * x.values + b
    plt.hlines(0, xl2[0], xl2[1], color='gray', linestyles='dashed')
    plt.vlines(0, yl2[0], yl2[1], color='gray', linestyles='dashed')
    #
    k = 0
    for n in name:
        plt.scatter(x[n], y[n], label=n, color=c[k], marker='^', s=2000)
        k += 1
        #
        #
    #
    plt.scatter(x['REAN'], y['REAN'], label=n, color='black', marker='^', s=4000)
    plt.plot(x.values, fit, ls=':', lw=1, color='black')
    plt.text(x['REAN'] + 0.01, y['REAN'] + 0.0005, 'REAN', fontsize=30, rotation=45)
    plt.text(-0.2, yl2[1] - 0.005, t + 'R$^2$=' + f'{r ** 2: .1%}' + ';', fontsize=30)
    plt.text(-0.2, yl2[1] - 0.015, ptext(pv), fontsize=30)
    plt.ylabel('PPT trend [mm day$^{-1}$ dec$^{-1}$]')
    plt.xlabel('SST trend [°C dec$^{-1}$]')
    plt.title(loc['pr'][l])
    plt.legend(bbox_to_anchor=(0.3, 0.1, 1.1, 1))
    plt.subplots_adjust(left=0.17, bottom=0.18, right=0.7)
    plt.ylim(yl2)
    plt.xlim(xl2)
    plt.savefig(loc['pr'][l] + '_' + p + '.pdf', dpi=300, format='pdf')
    # plt.show()
    plt.close()


# FIM
