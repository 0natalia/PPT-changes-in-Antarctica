# -*- coding: utf-8 -*-
'''
Precipitation changes in high southern latitudes

This script reads CMIP6 'tos' and 'pr' datasets as ensemble,
and calculates their ens. mean.
Then, for each defined Antarctic region [Austral, EAIS, Ross, Wed
and WAIS],
it estimates the variables' linear trend and other statistics.
At last, this script plots: PPT trend X SST trend, and their
correlation as R2.

Natália Silva (2021)
natalia3.silva@usp.br
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mscbib import norm_cmap, mod_agr  # ptext, my_cmap, clipper, linear_trend3
import seaborn as sns
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import pickle

# Figure setups
pd.set_option('display.float_format', lambda x: '%.2e' % x)
sns.set_context('talk', font_scale=1)  # large fontsize
plt.rcParams['hatch.linewidth'] = 0.3
plt.rcParams['hatch.color'] = 'gray'
plt.rcParams["figure.figsize"] = (10, 8)


# *_*_*_*_*_*_*_  PPT | SST trends   *_*_*_*_*_*_

lla = np.arange(-89.5, -39.5, 1)
llo = np.arange(0, 361, 1)
lon, lat = np.meshgrid(llo, lla)
cax = plt.axes([0.1, 0.1, 0.8, 0.1])
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.6  # 0.5 = 90-50 = 40°
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
map_proj = ccrs.Orthographic(0, -90.0)
map_transf = ccrs.PlateCarree()
grlat = [-90, -80, -70, -60, -50, -40]


# READING THE TRENDS
with open('trend_tos_mod.pkl', 'rb') as f:
    M_bef, M_aft = pickle.load(f)


with open('trend_tos_rean.pkl', 'rb') as f:
    R_bef, R_aft = pickle.load(f)


# Choose realization
var = R_aft.sel(stats='slope', realization=name[-1], drop='nan') - M_aft
name = var.realization
# tt = var.sel(stats='slope', realization=name[6])
# pp = var.sel(stats='pval', realization=name[6])

# Set plot limits according to the chosen variable
# plim = 0.1
# lev = np.linspace(-plim, plim, 50)
# cbtick = np.linspace(-plim, plim, 6)
# cmove = plt.cm.BrBG  # PuOr  # 
# midnorm = norm_cmap(vmin=-plim, vcenter=0, vmax=plim)
# cblab = r'$\Delta$PPT trend [mm day$^{-1}$ dec$^{-1}$]'
#
tlim = 0.5
lev = np.linspace(-tlim, tlim, 50)
cbtick = np.linspace(-tlim, tlim, 6)
cmove = plt.cm.RdGy_r  # RdBu_r  # 
midnorm = norm_cmap(vmin=-tlim, vcenter=0, vmax=tlim)
cblab = r'SST trend [$^o$C dec$^{-1}$]'
# $\Delta$

# Statistics on top of shading: model agreement on signal and trend signifiance
sig1 = pp.where(pp < 0.05)
# # Runs for Ens. means:
p = var.sel(stats='slope').where(var.sel(stats='slope') > 0).count('realization') >= mod_agr(len(name))
n = var.sel(stats='slope').where(var.sel(stats='slope') < 0).count('realization') >= mod_agr(len(name))

# PLOT
ax = plt.axes(projection=map_proj)
ax.coastlines(zorder=10)
ax.set_boundary(circle, transform=ax.transAxes)
# TREND VALUES SHADING
pl = plt.contourf(lon, lat, tt, lev, norm=midnorm, cmap=cmove,
                  transform=map_transf, extend='both', zorder=1)
plt.contour(lon, lat, tt, lev, norm=midnorm, cmap=cmove,
            transform=map_transf, extend='both', zorder=1)
# # MODEL AGREEMENT, AND SIGNIFICANT AT 1% and 5% IN OVERLAIED MARKERS
# plt.scatter(lon, lat, p, transform=map_transf,
#             zorder=10, color='gray', alpha=0.1, rasterized=True)
# plt.scatter(lon, lat, n, transform=map_transf,
#             zorder=10, color='gray', alpha=0.1, rasterized=True)
# ss = plt.contourf(lon, lat, sig1, transform=map_transf, colors='none',
#                   hatches='+', zorder=5)
# DRAW MAP PROPERTIES
cb = plt.colorbar(pl, shrink=0.8, spacing='proportional', label=cblab,
                  orientation="vertical", pad=0.1, ticks=cbtick)
cb.ax.set_yticklabels(["{:.2f}".format(i) for i in cbtick])
gl = ax.gridlines(linestyle='--', linewidth=1, zorder=3, color='lightgray')
gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl.ylocator = mticker.FixedLocator(grlat)
[plt.text(0, g, str(g), color='gray', fontsize=10, transform=map_transf) for g in grlat]
plt.subplots_adjust(bottom=0, top=0.95)
[c.set_rasterized(True) for c in pl.collections]
[s.set_rasterized(True) for s in ss.collections]
plt.show()
# plt.savefig('tosM' + var.period + '.pdf', dpi=100, format='pdf')
# plt.close()


# FIM
