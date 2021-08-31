# -*- coding: utf-8 -*-
'''
Precipitation changes in high southern latitudes

This script reads CMIP6 'tos' and 'pr' datasets as ensamble,
and calculates their ens. mean.
Then, for each defined Antarctic region [Austral, EAIS, Ross, Wed and WAIS],
it estimates the variables linear trend and other statistics.
At last, this script plots: PPT trend X SST trend, and their correlation as R2.

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
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import cartopy.crs as ccrs

# Figure setups
pd.set_option('display.float_format', lambda x: '%.2e' % x)
sns.set_context('talk', font_scale=1)  # large fontsize
plt.rcParams['hatch.linewidth'] = 0.3
plt.rcParams['hatch.color'] = 'gray'
plt.rcParams["figure.figsize"] = (10, 8)


# *_*_*_*_*_*_*_*_*_*_*   READ AND CONCATENATE DATA   _*_*_*_*_*_*_*_*_*_*
# *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_

# # ####### CMIP6 pr + tos #######
# path = ['/Volumes/snati/data/PPT/pmi4-cmip6/hist/*.nc',
#         '/Volumes/snati/data/SST/cmip6_hist/*.nc']
# m = {'pr': [], 'tos': []}

# for p, n in zip(path, m):
#     files = sorted(glob.glob(p))
#     dset, name = [], []
#     for f in files:
#         ds = xr.open_dataset(f, drop_variables=[
#                              'time_bnds', 'initial_time0_encoded'])
#         ds = ds.assign_coords({'realization': ds.source_id})
#         ds = ds.sel(lat=slice(-89.5, -40.5), time=slice('1900', '2014'))
#         ds['time'] = ds.time.dt.year
#         dset.append(ds)
#         name.append(ds.source_id)
#         #
#         #
#     m[n] = xr.concat(dset, 'realization')
#     m[n] = xr.concat([m[n], m[n].mean('realization').assign_coords(
#         {'realization': 'Ens.M'})], dim='realization')
#     #


# ####### Reanalysis #######
path = ['/Volumes/snati/data/PPT/reanalise/pr*.nc',
        '/Volumes/snati/data/SST/obs-rean/sst*.nc',
        '/Volumes/snati/data/WND/*vwnd.nc']
rname = ['20thCv3', 'CFSR', 'Era20C', 'EraInt', 'JRA55', 'MERRA2', 'NCEP2']
rean = {'pr': [], 'tos': [], 'vwnd': []}

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


# rean['tos'].mean(['lat','lon']).vwnd.plot(hue='realization')

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


# *_*_*_*_*_*_*_ PPT vs SST CORRELATION (r)   _*_*_*_*_*_*_*_*_*_
# *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*

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
# Set plot limits
plim = 1
lev = np.linspace(-plim, plim, 50)
cbtick = np.linspace(-plim, plim, 6)
cmove = plt.cm.Spectral_r
midnorm = norm_cmap(vmin=-plim, vcenter=0, vmax=plim)
cblab = 'r (Pearson)'

loc = ['Austral', 'EAIS', 'W-WAIS_Ross', 'E-WAIS_Amundsen', 'Weddell Sea']
i = 4  # range(0, len(loc))
reg, _, _ = clipper(rean['pr'].pr, loc[i])  # enter xr.DataArray
reg = reg.mean(['lat', 'lon'])

corr = xr.corr(reg, rean['tos'].tos, dim='time')
corr = xr.concat([corr, corr[:, :, -1]], dim='lon')
m = corr.where(corr > 0).count('realization') >= mod_agr(len(rname))
n = corr.where(corr < 0).count('realization') >= mod_agr(len(rname))


ax = plt.axes(projection=map_proj)
ax.coastlines(zorder=10)
ax.set_boundary(circle, transform=ax.transAxes)
# CORR VALUES SHADING
pl = plt.contourf(lon, lat, corr.sel(realization='REAN'), lev, norm=midnorm,
                  cmap=cmove, transform=map_transf, extend='both', zorder=1)
plt.contour(lon, lat, corr.sel(realization='REAN'), lev, norm=midnorm,
            cmap=cmove, transform=map_transf, extend='both', zorder=1)
# MODEL AGREEMENT, AND SIGNIFICANT AT 1% and 5% IN OVERLAIED MARKERS
plt.scatter(lon, lat, m, transform=map_transf,
            zorder=10, color='black', alpha=0.1, rasterized=True)
plt.scatter(lon, lat, n, transform=map_transf,
            zorder=10, color='black', alpha=0.1, rasterized=True)
# DRAW MAP PROPERTIES
CS = ax.contour(lon, lat, corr.sel(realization='REAN'), 5, colors='k',
                transform=map_transf, zorder=10, alpha=0.4)
ax.clabel(CS, inline=True, fontsize=10)
cb = plt.colorbar(pl, shrink=0.8, spacing='proportional', label=cblab,
                  orientation="vertical", pad=0.1, ticks=cbtick)
cb.ax.set_yticklabels(["{:.2f}".format(i) for i in cbtick])
gl = ax.gridlines(linestyle='--', linewidth=1, zorder=3, color='lightgray')
gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl.ylocator = mticker.FixedLocator(grlat)
[plt.text(0, g, str(g), color='gray', fontsize=10, transform=map_transf) for g in grlat]
[c.set_rasterized(True) for c in pl.collections]
[S.set_rasterized(True) for S in CS.collections]
# plt.show()
# plt.savefig('corr_ppt_TOS_' + loc[i] + '.pdf', dpi=100, format='pdf')
plt.close()


# FIM
