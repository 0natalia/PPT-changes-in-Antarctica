# -*- coding: utf-8 -*-
'''
Precipitation changes in high southern latitudes

This script reads CMIP6 and reanalyses datasets as an ensemble
and calculates their ens. mean.
Then, for Antarctic regions [Austral, EAIS, Ross, Wed, and WAIS
cut using clipper function in mscbib], it estimates the precipitation
linear trend.
At last, this script plots: (i) regional precipitation time series
                            (ii) regional ppt anomalies relative to 1979:2010
                            (iii) calculated trends for each dataset
                                  as barplot
                            (iv) linear fit time series
                            (v) ppt trend per dataset as a heatmap

Note: always calculate reanalyses-ensemble mean and model-ensemble
mean separately. Do not mix different data types altogether.

Natália Silva (2021)
natalia3.silva@usp.br
'''

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from mscbib import clipper, ptext, my_cmap, tinv
import glob
import seaborn as sns
from scipy import stats
from xclim import ensembles as ens
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.float_format', lambda x: '%.2e' % x)
c = my_cmap(25, 'm')

# *_*_*_*_*_*_*_*_*_*_*   READ AND CONCATENATE DATA   _*_*_*_*_*_*_*_*_*_*
# *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_

path = ['/Volumes/snati/data/PPT/cmip6_hist/amip/*.nc',
        '/Volumes/snati/data/PPT/cmip6_hist/hist/*.nc']

for p in path:
    files = sorted(glob.glob(p))
    dset, name = [], []
    for f in files:
        ds = xr.open_dataset(f)
        ds = ds.assign_coords({'realization': ds.source_id})
        ds = ds.sel(lat=slice(-89.5, -40.5), time=slice('1900', '2014'))
        ds['time'] = ds.time.dt.year
        dset.append(ds)
        name.append(ds.source_id)
        #
        #
    hist = xr.concat(dset, 'realization')
    # d[n] = xr.concat([d[n], d[n].mean('realization').assign_coords({'realization': 'Ens.M'})], dim='realization')
    #


# ####### Reanalysis #######
path = ['/Volumes/snati/data/PPT/reanalise/pr*.nc']
rname = ['20thCv3', 'CFSR', 'Era20C', 'EraInt', 'JRA55', 'MERRA2', 'NCEP2']

for p in path:
    files = sorted(glob.glob(p))
    dset = []
    for f in files:
        ds = xr.open_dataset(f, drop_variables=[
                             'time_bnds', 'initial_time0_encoded'])
        ds = ds.sel(lat=slice(-89.5, -40.5), time=slice('1900', '2014'))
        ds['time'] = ds.time.dt.year
        dset.append(ds)
        #
        #
    rean = xr.concat(dset, 'realization').assign_coords({'realization': rname})
    # rean[n] = xr.concat([rean[n], rean[n].mean('realization').assign_coords(
    #     {'realization': 'REns.M'})], dim='realization')


# del(files, f, ds, dset, glob, rean, p, path)

# *_*_*_*_*_*_*_ CALCULATE TREND PER REGION PER DATA   _*_*_*_*_*_*_*_*_*_
# *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_
# Regional time series

loc = ['Austral', 'EAIS', 'W-WAIS_Ross', 'E-WAIS_Amundsen', 'Weddell Sea']
i = 4  # range(0, len(loc))

reg, _, _ = clipper(rean.pr, loc[i])  # enter xr.DataArray
reg = reg.mean(['lat', 'lon'])

ens_stats = ens.ensemble_mean_std_max_min(reg.to_dataset())  # xr.Dataset

dts = reg.T.to_pandas()
dts['Ens.M'] = reg.mean('realization', skipna=True).to_pandas()

# # If dealing with CMIP6 hist, compare with the reanalyses ens.m
# reg, _, _ = clipper(rean.pr, loc[i])  # enter xr.DataArray
# reg = reg.mean(['lat', 'lon']).to_pandas().T
# dts['Obs/R'] = reg.mean(1)


# Empty dset to be filled with statistics results for each dataset
ts = tinv(0.05, len(dts.index) - 2)  # 95% confidence interval on slope
res = pd.DataFrame(columns=dts.columns, index=[
                   'slpB80', 'bB80', 'rB80', 'pB80', 'errB80',
                   'slpA80', 'bA80', 'rA80', 'pA80', 'errA80'])
for column in dts:
    qi = dts[column].loc[dts.index[0]:1979].dropna()
    if qi.empty:
        res[column][5] = np.nan
    else:
        res[column][0:5] = stats.linregress(qi.index, qi)
    qf = dts[column].loc[1980:2014].dropna()
    res[column][5:10] = stats.linregress(qf.index, qf)
    #


# slp, intercept, correlation coef, pval, slope error, intercept error
# All-time:
# qq = dts[column].dropna()
# res[column][0:5] = stats.linregress(qq.index, qq)
# print(f"{loc[i], column} slopeB (95%):
#       {res[column]['slpB80'] * 10:.2e} +/-
#       {ts * res[column]['errB80'] * 10:.2e}")
# print(f"{loc[i], column} slopeA (95%):
#       {res[column]['slpA80'] * 10:.2e} +/-
#       {ts * res[column]['errA80'] * 10:.2e}")
# print(f"{loc[i], column} slopeB: {res[column]['slpB80'] * 10:.3f}")
# print(f"{loc[i], column} pB: {res[column]['pB80']:.2e}")
# print(f"{loc[i], column} slopeA: {res[column]['slpA80'] * 10:.3e}")
# print(f"{loc[i], column} pA: {res[column]['pA80']:.2e}")


# Empty 'table' to be filled with linear fit for each dataset
# Note: each dataset may cover different periods (only for reanalysis)
# fit = pd.DataFrame(columns=dts.columns, index=d.time.values)
# for col in dts:
#     ti = dts[col].first_valid_index()
#     tf = dts[col].last_valid_index()
#     ii = dts.index.get_loc(ti)
#     i_f = dts.index.get_loc(tf) + 1
#     fit[col][ii:t80] = res[col]['slpB80'] *\
#         np.arange(ti, 1979 + 1, 1) + res[col]['bB80']  # fit = a * x + b
#     fit[col][t80:i_f] = res[col]['slpA80'] *\
#         np.arange(1980, tf + 1, 1) + res[col]['bA80']  # fit = a * x + b


del(qf, qi, t80, col, column, ti, tf, ii, i_f)


# *_*_*_*_*_*_*_*_*_*_*_*_*_   PLOT    _*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_
# *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_
# Trend value per decade # mm/day/dec
res.loc['slpB80'] *= 10
res.loc['slpA80'] *= 10
pB80 = ptext(res['Ens.M']['pB80'])
pA80 = ptext(res['Ens.M']['pA80'])


# *_*_*_*_*_*_*_*_*_*_*_*_*_
# Trends per dataset according to its significance plot
# Trend values significative at 1% level dataframe
t1 = pd.DataFrame(columns=dts.columns)
t1.loc['bef 80'] = res.loc['slpB80'].where(res.loc['pB80'] <= 0.01)
t1.loc['aft 80'] = res.loc['slpA80'].where(res.loc['pA80'] <= 0.01)

# Trend values significative at 5% level dataframe
t5 = pd.DataFrame(columns=dts.columns)
t5.loc['bef 80'] = res.loc['slpB80'].where(res.loc['pB80'].between(0.01, 0.05))
t5.loc['aft 80'] = res.loc['slpA80'].where(res.loc['pA80'].between(0.01, 0.05))

# Trend values with p values higher than 0.05
tt = pd.DataFrame(columns=dts.columns)
tt.loc['bef 80'] = res.loc['slpB80'].where(res.loc['pB80'] > 0.05)
tt.loc['aft 80'] = res.loc['slpA80'].where(res.loc['pA80'] > 0.05)

plt.style.use('seaborn-white')
sns.set_context('talk', font_scale=0.8)
plt.figure(figsize=(15, 4))
# sns.heatmap(t1, annot=True, cmap='BrBG', vmin=-5e-2, vmax=5e-2, annot_kws={'size': 17, 'weight': 'bold', 'rotation': 45}, fmt='.2e', cbar=False)
tp = sns.heatmap(t5, annot=True, cmap='BrBG', vmin=-5e-2, vmax=5e-2, annot_kws={'size': 17, 'style': 'italic', 'rotation': 45}, fmt='.2e', cbar_kws={'extend': 'both', 'label': 'PPT trend [mm day$^{-1}$ dec$^{-1}$]'})
sns.heatmap(tt, annot=True, cmap='BrBG', vmin=-5e-2, vmax=5e-2, fmt='.2e', annot_kws={'size': 13, 'rotation': 45}, cbar=False)
tp.set_title(loc[i] + ' PPT trends before and after 1979 [mm day$^{-1}$ dec$^{-1}$]')
plt.subplots_adjust(left=0.03, bottom=0.4, right=0.7)
plt.xticks(rotation=55)
# plt.show()
plt.savefig('heatmap' + loc[i] + '.pdf', dpi=70, facecolor='w', format='pdf')
plt.close()


# *_*_*_*_*_*_*_*_*_*_*_*_*_
# Trend bar plot

# # print(plt.style.available)
# plt.style.use('seaborn-darkgrid')
# plt.rcParams["figure.figsize"] = (14, 4)

# sns.set_context('talk', font_scale=1)
# gridval = [-1e-1, -1e-2, -1e-3, -1e-4, 0, 1e-4, 1e-3, 1e-2, 1e-1]
# ll = len(dts.columns)
# ylab = list(map('{:.0e}'.format, gridval))
# plt.bar(np.arange(ll) * 3 + 0.5, res.iloc[0], color=c)
# # , yerr=ts * res.iloc[4] * 10, ecolor='gray'
# plt.bar(np.arange(ll) * 3 + 1.5, res.iloc[5], color=c)
# # , yerr=ts * res.iloc[9] * 10, ecolor='gray'
# plt.xticks(np.arange(ll) * 3 + 1, list(dts.columns), rotation=45)
# plt.ylabel(f'PPT trend [mm day$^-$$^1$ dec$^-$$^1$]')
# plt.yscale('symlog', linthreshy=1e-4)
# plt.ylim(-5e-1, 5e-1)
# plt.title(loc[i])
# plt.yticks(gridval, ylab)
# plt.subplots_adjust(bottom=0.3)
# # plt.show()
# plt.savefig('trendBAR_' + loc[i] + '.pdf', dpi=50, facecolor='w', format='pdf')
# plt.close()


# *_*_*_*_*_*_*_*_*_*_*_*_*_
# Série

# z = 1.96 --> IC = 95% --> p_val = 0.05
# multiplicando o desvio padrão por 1.96, vamos encontrar um
# intervalo que abrange 95% dos dados ao redor da média.
pr5 = ens_stats.pr_mean - ens_stats.pr_stdev * 1.96
pr95 = ens_stats.pr_mean + ens_stats.pr_stdev * 1.96
dr = dts.rolling(2, center=True, min_periods=2).mean()
pr5 = pr5.rolling(time=2, center=True, min_periods=2).mean()
pr95 = pr95.rolling(time=2, center=True, min_periods=2).mean()

# print(plt.style.available)
plt.style.use('seaborn-darkgrid')
plt.rcParams["figure.figsize"] = (14, 4)

sns.set_context('talk', font_scale=0.85)
ax = dr.plot(color=c, linewidth=1, alpha=1, zorder=1)
dr['Ens.M'].plot(linewidth=5, color='black', zorder=5)
plt.plot(dr.index, dr['Obs/R'], linewidth=5, color='black', linestyle='-.',
         zorder=5, label='Obs/R ens.m')
plt.fill_between(dts.index, pr5, pr95, color='gray', edgecolor='gray',
                 alpha=0.1, label='CI: 95%', zorder=2)
plt.subplots_adjust(left=0.07, bottom=0.15, right=0.8)
plt.xlabel('yr')
plt.ylabel('PPT [mm day$^{-1}$]')
lim = ax.get_ylim()
plt.text(dr.index[5], 1.28, 'b80: ' + f"{res['Ens.M']['slpB80']: .2e} +/- {ts * res['Ens.M']['errB80'] * 10: .2e}" + '; ' + pB80)
plt.text(dr.index[5], 1.2, 'a80: ' + f"{res['Ens.M']['slpA80']: .2e} +/- {ts * res['Ens.M']['errA80'] * 10: .2e}" + '; ' + pA80)
plt.title(loc[i])
plt.ylim(lim)
plt.xlim(dts.index[0], dts.index[-1])
ax.legend(bbox_to_anchor=(0.25, 0.1, 1, 1))
# plt.show()
plt.savefig('MODserie' + loc[i] + '.pdf', dpi=50, facecolor='w', format='pdf')
plt.close()


# *_*_*_*_*_*_*_*_*_*_*_*_*_
# Anomalie time series plot
# anom = dts - dts.loc[1979:2010].mean()
# dr = anom.rolling(2, center=True, min_periods=2).mean()

# ax = dr.plot(color=c, linewidth=0.7, alpha=1, zorder=2)
# dr['Ens.M'].plot(linewidth=5, color='black', zorder=5)
# plt.plot(dr.index, dr['Obs/R'], linewidth=3, color='black', linestyle='--',
#          zorder=5, label='Obs/R ens.m')
# lim = ax.get_ylim()
# plt.subplots_adjust(left=0.08, bottom=0.15, right=0.8)
# plt.xlabel('yr')
# plt.ylabel('Δ PPT [mm day$^{-1}$]')
# plt.ylim(lim)
# plt.title(loc[i])
# plt.xlim(dts.index[0], dts.index[-1])
# ax.axvspan(1979, 2010, alpha=0.3, color='lightgray', hatch='/', label='1979:2010')
# ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], alpha=0.4, color='gray')
# ax.legend(bbox_to_anchor=(0.25, 0.1, 1, 1))
# plt.show()
# # plt.savefig('MODanom' + loc[i] + '.pdf', dpi=50, facecolor='w', format='pdf')
# plt.close()


# *_*_*_*_*_*_*_*_*_*_*_*_*_
# # Fitted by linear regression plot
# plt.style.use('seaborn-white')
# plt.rcParams["figure.figsize"] = (10, 4)
# ax = fit.plot(color=c, linewidth=2, alpha=1, zorder=3)
# dts.rolling(3, center=True, min_periods=2).mean().plot(ax=ax, color=c, linewidth=1, alpha=0.15, zorder=1)
# ax.legend(bbox_to_anchor=(1, 1))
# plt.subplots_adjust(left=0.09, bottom=0.15, right=0.75)
# plt.xlim(dts.index[0], dts.index[-1])
# plt.xlabel('yr')
# plt.ylabel('ppt [mm day$^{-1}$]')
# plt.title(loc[i])
# plt.legend(bbox_to_anchor=(1, 1))
# # plt.ylim(lim)
# plt.show()
# # plt.savefig('fit_' + loc[0] + '.pdf', dpi=300, facecolor='w', format='pdf')
# plt.close()


del(pr5, pr95, dts, res, fit, ax, c, gridval,
    i, my_cmap, ptext, ylab, pA80, pB80)


# FIM
