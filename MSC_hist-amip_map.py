# -*- coding: utf-8 -*-
'''
Precipitation changes in high southern latitudes

This script reads sets of CMIP6 hist and amip simulations as ensemble,
and calculates the ensemble mean.
It also calculates the difference in precipitation between the
experiments (historical @minus amip = possible ocean influence on ppt)
Here, we estimate the difference for the mean period 1979-2014, common
for both experiments.
At last, this script plots: (i) map of the difference between the datasets

Model agreement: stippling on the maps indicate model agreement in
the sign of the difference, at the 95% significance level.

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
from mscbib import norm_cmap, mod_agr
from xclim import ensembles as ens
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (10, 8)
sns.set_context('talk')  # large fontsize


# *_*_*_*_*_*_*_*_*_*_*   READ AND CONCATENATE DATA   _*_*_*_*_*_*_*_*_*_*
# *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_

path = ['/Volumes/snati/data/PPT/cmip6_hist/amip/*.nc',
        '/Volumes/snati/data/PPT/cmip6_hist/hist/*.nc']
d = {'amip': [], 'hist': []}

for p, n in zip(path, d):
    files = sorted(glob.glob(p))
    dset, name = [], []
    for f in files:
        ds = xr.open_dataset(f)
        ds = ds.assign_coords({'realization': ds.source_id})
        ds = ds.sel(lat=slice(-89.5, -40.5), time=slice('1979', '2014'))
        ds['time'] = ds.time.dt.year
        dset.append(ds)
        name.append(ds.source_id)
        #
        #
    d[n] = xr.concat(dset, 'realization')
    #
    #

amip, hist = list(d.values())[0], list(d.values())[1]


del(files, f, ds, dset, glob, d, n, p, path)


# *_*_*_*_*_*_*_*_*_*_*_*   HIST - AMIP = OCN   *_*_*_*_*_*_*_*_*_*_*_*_*_
# *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_

diff = amip - hist

# *_*_*_*_*_*_*_*_*_*_*_*_*   MODELS MAP PLOT   *_*_*_*_*_*_*_*_*_*_*_*_*_
# *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_

diff = xr.concat((diff.pr, diff.pr[:, :, :, -1:]), dim='lon').mean('time')

diff = diff.to_dataset()
ens_stats = ens.ensemble_mean_std_max_min(diff)

p = diff.pr.where(diff.pr > 0).count('realization') >= mod_agr(len(name))
n = diff.pr.where(diff.pr < 0).count('realization') >= mod_agr(len(name))

name = name + ['Ens.M']
ensm = ens_stats.pr_mean.assign_coords({'realization': 'Ens.M'})
diff = xr.concat([diff.pr, ensm], dim='realization')

lla, llo = np.arange(-89.5, -39.5, 1), np.arange(0, 361, 1)
lon, lat = np.meshgrid(llo, lla)
cax, theta = plt.axes([0.1, 0.1, 0.8, 0.1]), np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5  # 0.5 = 90-50 = 40°
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
map_proj, map_transf = ccrs.Orthographic(0, -90.0), ccrs.PlateCarree()
grlat = [-90, -80, -70, -60, -50, -40]

lev, cbtick = np.linspace(-0.3, 0.3, 100), np.linspace(-0.3, 0.3, 11)
midnorm = norm_cmap(vmin=-0.3, vcenter=0, vmax=0.3)
cmove = plt.cm.RdGy  # cm.make_cmap('ColorMoves/4w_ROTB.xml', 'flip')
cmove = cmove.reversed()

# Change r for each dataset + ensamble mean (loop in 'realization' dim)
r = name[-1]

ax = plt.axes(projection=map_proj)
ax.set_boundary(circle, transform=ax.transAxes)
plt.scatter(lon, lat, p, transform=map_transf,
            zorder=10, color='black', alpha=0.15)
plt.scatter(lon, lat, n, transform=map_transf,
            zorder=10, color='black', alpha=0.15)
rr = plt.contourf(lon, lat, diff.sel(realization=r), lev, norm=midnorm,
                  cmap=cmove, transform=map_transf, extend='both')
plt.contour(lon, lat, diff.sel(realization=r), lev, norm=midnorm,
            cmap=cmove, transform=map_transf, extend='both')
CS = ax.contour(lon, lat, diff.sel(realization=r), 16, colors='k',
                transform=map_transf, zorder=10)
ax.clabel(CS, inline=True, fontsize=10)
cb = plt.colorbar(rr, shrink=0.8, spacing='proportional',
                  orientation="vertical", pad=0.02)
cb.set_ticks(cbtick)
cb.set_label('mm/day')
ax.coastlines(zorder=2, color='gray')
ax.set_title(r + ' hist - amip = ocn')
gl = ax.gridlines(linestyle='--', linewidth=1, zorder=3, color='lightgray')
gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl.ylocator = mticker.FixedLocator(grlat)
[plt.text(0, g, str(g), color='gray', fontsize=10, transform=map_transf) for g in grlat]
plt.subplots_adjust(bottom=0, top=0.95)
# plt.show()
plt.savefig(r + '.pdf', dpi=300, format='pdf')
plt.close()


# FIM
