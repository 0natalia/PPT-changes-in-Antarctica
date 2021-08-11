# -*- coding: utf-8 -*-
'''
Precipitation changes in high southern latitudes

This script reads reanalyses and CMIP6 datasets as ensemble,
and calculates their ensemble mean.
It also addresses the mean PPT field south of 45• and the difference
of each dataset from the ens. mean.
In this script, the ens. mean map is overlaped with the models agreement
as stippling - model agreement is defined by the ensemble size through
binomial distribution. For achieving 95% confidence level on agreement, 6/7
reanalyses and 15/23 models must agree.
At last, this script plots: (i) map of all-time average ppt for each dataset
                            (ii) map of the 50yr difference for each dset


Natália Silva (2021)
natalia3.silva@usp.br
'''

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import cartopy.crs as ccrs
import glob
import xarray as xr
import cm_xml_to_matplotlib as cm
from mscbib import norm_cmap, mod_agr
from xclim import ensembles as ens
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")


plt.rcParams["figure.figsize"] = (10, 6)
sns.set_context('talk')  # large fontsize


# *_*_*_*_*_*_*_*_*_*_*   READ AND CONCATENATE DATA   _*_*_*_*_*_*_*_*_*_*
# *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_

# ####### CMIP6 pr + tos #######
path = '/Volumes/snati/data/PPT/cmip6_hist/hist/*.nc'
files = sorted(glob.glob(path))
dset, name = [], []
for f in files:
    ds = xr.open_dataset(f, drop_variables=[
                         'time_bnds', 'initial_time0_encoded'])
    ds = ds.assign_coords({'realization': ds.source_id})
    ds = ds.sel(lat=slice(-89.5, -40.5), time=slice('1900', '2014'))
    ds['time'] = ds.time.dt.year
    dset.append(ds)
    name.append(ds.source_id)


m = xr.concat(dset, 'realization')
m = xr.concat([m, m.mean('realization').assign_coords(
    {'realization': 'CMIP6'})], dim='realization')


# ####### Reanalysis #######
path = '/Volumes/snati/data/PPT/reanalise/pr*.nc'
rname = ['20thCv3', 'CFSR', 'Era20C', 'EraInt', 'JRA55', 'MERRA2', 'NCEP2']
files = sorted(glob.glob(path))
dset = []
for f in files:
    ds = xr.open_dataset(f, drop_variables=[
                         'time_bnds', 'initial_time0_encoded'])
    ds = ds.sel(lat=slice(-89.5, -40.5), time=slice('1900', '2014'))
    ds['time'] = ds.time.dt.year
    dset.append(ds)


rean = xr.concat(dset, 'realization').assign_coords({'realization': rname})
rean = xr.concat([rean, rean.mean('realization').assign_coords(
    {'realization': 'REAN'})], dim='realization')


del(files, f, ds, dset, glob)


# *_*_*_*_*_*_*_*_*_*_*_*_*   MODELS MAP PLOT   *_*_*_*_*_*_*_*_*_*_*_*_*_
# *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_

# All-time average -- squeeze time: 3D matriz = [lat, lon, rean]
# concatenar em 360°
d = m
ppt = xr.concat((d.pr, d.pr[:, :, :, -1:]), dim='lon').mean('time')


# Model agreement according to the mod_agr function (mscbib Python library)
# For our purposes, models must agree within the interval mean +- 1 std
ppt = ppt.to_dataset()
ens_stats = ens.ensemble_mean_std_max_min(ppt)
pr5 = ens_stats.pr_mean - ens_stats.pr_stdev * 1
pr95 = ens_stats.pr_mean + ens_stats.pr_stdev * 1
a = pr5 <= ppt
b = ppt <= pr95
s = a * b
sig = s.where(s.pr==True).count('realization')
agr = sig.pr >= mod_agr(len(ppt.realization.values) - 1)

name = ppt.realization.values

lla = np.arange(-89.5, -39.5, 1)
llo = np.arange(0, 361, 1)
lon, lat = np.meshgrid(llo, lla)
cax = plt.axes([0.1, 0.1, 0.8, 0.1])
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5  # 0.5 = 90-50 = 40°
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
map_proj = ccrs.Orthographic(0, -90.0)
map_transf = ccrs.PlateCarree()
grlat = [-90, -80, -70, -60, -50, -40]

idx = ['avg', 'ens_diff']
lev = [np.linspace(0, 8, 200), np.linspace(-1.5, 1.5, 100)]
cbtick = [np.linspace(0, 8, 9), np.linspace(-1.5, 1.5, 11)]
cmove = [cm.make_cmap('ColorMoves/other-outl-1.xml', 'flip'), plt.cm.BrBG]
# cm.make_cmap('ColorMoves/4w_ROTB.xml','flip')]  plt.cm.PuBu,
midnorm = [norm_cmap(vmin=0, vcenter=3.5, vmax=8),
           norm_cmap(vmin=-1.5, vcenter=0, vmax=1.5)]
ext = ['max', 'both']
cblab = [r'PPT [mm day$^{-1}$]', r'SST [mm day$^{-1}$]']

# Change r for each dataset + ensamble mean (loop in 'realization' dim)
r = name[-1]
# ens_diff = ppt.sel(realization='CMIP6') - ppt.sel(realization=r)
# var = xr.Dataset(
#     {"avg": (["lat", "lon"], ppt.sel(realization=r)),
#         "ens_diff": (["lat", "lon"], ens_diff), },
#     coords={"lat": lla, "lon": llo, }, )
# tit = ['ppt mean from 1850 to 2014 -- ' + r,
#        'difference of the ens. mean to ' + r]
i = 0
ax = plt.axes(projection=map_proj)
ax.set_boundary(circle, transform=ax.transAxes)
plt.scatter(lon, lat, agr, transform=map_transf,
            zorder=10, color='black', alpha=0.15, rasterized=True)
pl = plt.contourf(lon, lat, ppt.sel(realization=r).pr, lev[i], norm=midnorm[i],
                  cmap=cmove[i], transform=map_transf, extend=ext[i])
pp = plt.contour(lon, lat, ppt.sel(realization=r).pr, lev[i], norm=midnorm[i],
                 cmap=plt.get_cmap(cmove[i]), transform=map_transf,
                 extend=ext[i])
cb = plt.colorbar(pl, shrink=0.8, spacing='proportional',
                  orientation="vertical", pad=0.02)
cb.set_ticks(cbtick[i])
cb.set_label(cblab[i])
ax.coastlines(zorder=2)
ax.set_title(r)
gl = ax.gridlines(linestyle='--', linewidth=1, zorder=3, color='lightgray')
gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl.ylocator = mticker.FixedLocator(grlat)
[plt.text(0, g, str(g), color='gray', fontsize=10, transform=map_transf) for g in grlat]
plt.subplots_adjust(bottom=0, top=0.95)
[l.set_rasterized(True) for l in pl.collections]
[p.set_rasterized(True) for p in pp.collections]
[c.set_rasterized(True) for c in cb.collections]
plt.show()
# plt.savefig(idx[i] + r + '.pdf', dpi=50, format='pdf')
plt.close()
# print(r)


# FIM
