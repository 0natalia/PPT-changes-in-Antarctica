# -*- coding: utf-8 -*-
'''
Function that cuts/clips Antarctic regions
Using XArray and RioXArray

Natália Silva (2020)
'''

import rioxarray
import xarray as xr
import numpy as np
from scipy.stats import t, linregress
import matplotlib as mpl


def linear_trend3D(x):
    '''
    Input: x = [time, lat, lon] Dataset
    Output: [lat, lon] trend Dataset, with two variables in 'stats' dimension:
    slope and its p_value
    Calculates linear trend for a 3D DataSet (usually [time, lat, lon]).
    It is necessary to 'stack' [lon,lat] as 'allpoints'in order to perform
    the analysis.
    Onde the function linear_trend3D is applied, one might 'unstack'
    and obtain the [lon,lat] coordinates again.
    '''
    slp, _, _, pv, _ = linregress(x.time, x)
    # need to return an xr.DataArray for groupby
    return xr.DataArray([slp, pv])


def tinv(p, df):
    '''
    Two-sided inverse Students t-distribution
    p - probability, df - degrees of freedom
    From:https://docs.scipy.org/doc/scipy/reference/generated
    /scipy.stats.linregress.html

    '''
    return abs(t.ppf(p / 2, df))


def mod_agr(N):
    '''
    Model agreement:
        how many datasets must agree -- according to criteria --
        to have a statistical (95%) significative result

    Ex: plot curves with different N
    # plt.figure(1,figsize=(12,7))
    # plt.plot(x, bi, '-o', ms=8, label='Obs/R: n = 9')
    # plt.plot(y, bi2, '-x', ms=8, label='hist: n = 23')
    # plt.ylim([0, 1.1])
    # plt.xlim([0.5, n + 0.5])
    # plt.yticks((np.arange(0, 1.1, 0.1)),fontsize=18)
    # plt.xticks(y, fontsize=18)
    # plt.axhline(y=0.95, color='r', lw=2)
    # plt.ylabel('probability', fontsize=18)
    # plt.xlabel('Model Agreement', fontsize=18)
    # plt.legend()
    # plt.grid()
    # plt.show()
    # plt.close()

    '''
    from scipy.stats import binom
    x = np.arange(1, N + 1)
    bi = binom.cdf(x, N, 0.5)
    return(np.where(bi > 0.95)[0][0] + 1)


def ptext(p_val):
    '''
    Format p_value as text style.
    Calculate from stats.lineregress()

    '''
    if p_val < 0.01:
        p_val = 'p < 0.01'
    elif p_val > 0.01 and p_val < 0.05:
        p_val = 'p < 0.05'
    elif p_val > 0.05 and p_val < 0.1:
        p_val = 'p < 0.1'
    elif p_val > 0.1:
        p_val = 'p > 0.1'
    return(p_val)


class norm_cmap(mpl.colors.Normalize):
    '''
    Normaliza colormaps p/ melhor distribuílas.
    Exiva perca de resolução por cores caso haje pontos "outliers"
    Ex: média de ptt na Antártica=1 mm/dia, mas no sul do chive ppt=11 mm/dia
    Se aumentar a colorbar p/ 11 perde resolução.

    '''

    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def my_cmap(n, d):
    '''
    Create a colorbase based on #RBG colors. Particurlary usefull to plot
    reanalysis ensamble.

    Test plot:
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

x = np.linspace(0, 0.5 * np.pi, 64)
y = np.cos(x)
c = rean_cmap(np.arange(24))

plt.figure()
plt.plot(x,y)
n = len(c)
for i in range(n):
    plt.plot(x, i*y, color=c[i], linewidth=3)

plt.show()
plt.close()

    More hex color codes in: https://www.color-hex.com/
    '''
    if d == 'R':
        cmap = ['#34A048', '#EC008C', '#00AEEF',
                '#AD71B5', '#FDBF6E', '#7fccce', '#f45a49', '#000000',
                '#99c084']  # '#896bdd', '#F57E20', 
    else:
        cmap = ['#E21F26', '#00ff00', '#5F98C6',
                '#60C8E8', '#FDBF6E', '#EC008C', '#F799D1',
                '#fbd266', '#34A048', '#AD71B5', '#d0965b', '#63ffbc',
                '#ff9292', '#00AEEF', '#F57E20', '#36ff94', '#B35B28',
                '#FFD700', '#7fccce', '#f45a49', '#896bdd',
                '#ff4ff8', '#1189f3', '#000000', '#A3A09F']
    rean_cmap = mpl.colors.ListedColormap(cmap, name='rean_cmap')
    return rean_cmap(np.arange(n))


def clipper(var, reg):
    '''
    Clipps/cuts any polygonal region of var(lat/lon,time) Data Array.
    "Clipper" function, specifically, cuts matrixes from the Austral region

    Note: to define a region, input the vertices as [lon (0:360), lat]
          la = [southernmost lat + 0.5, northernmost lat + 0.5]
          lo = [menor + 1, maior + 1]

    '''
    var.rio.set_crs("epsg:4326")
    if reg == 'Austral':
        geo = [{'type': 'Polygon', 'coordinates': [
            [[0, -60], [360, -60], [360, -91], [0, -91]]]}]
        la = np.arange(-89.5, -59.5, 1)
        lo = np.arange(1, 360, 1)
    elif reg == 'EAIS':
        geo = [{'type': 'Polygon', 'coordinates': [
            [[29, -87], [29, -75], [141, -75], [141, -87]]]}]
        la = np.arange(-86.5, -74.5, 1)
        lo = np.arange(30, 142, 1)
    elif reg == 'E-WAIS_Amundsen':
        geo = [{'type': 'Polygon', 'coordinates': [
            [[280, -75], [280, -85], [235, -85], [235, -75]]]}]
        la = np.arange(-84.5, -74.5, 1)
        lo = np.arange(236, 281, 1)
    elif reg == 'W-WAIS_Ross':
        geo = [{'type': 'Polygon', 'coordinates': [
            [[225, -83], [170, -83], [170, -75], [225, -75]]]}]
        la = np.arange(-82.5, -74.5, 1)
        lo = np.arange(171, 226, 1)
    elif reg == 'Weddell Sea':
        geo = [{'type': 'Polygon', 'coordinates': [
            [[315, -80], [295, -80], [304, -68], [345, -68]]]}]
        la = np.arange(-79.5, -67.5, 1)
        lo = np.arange(296, 344, 1)
    elif reg == 'Indian Oc. Sector':
        geo = [{'type': 'Polygon', 'coordinates': [
            [[53, -65], [53, -50], [145, -50], [145, -65]]]}]
        la = np.arange(-64.5, -49.5, 1)
        lo = np.arange(54, 146, 1)
    elif reg == 'Amundsen-Bellingshausen':
        geo = [{'type': 'Polygon', 'coordinates': [
            [[291, -60], [282, -71], [216, -73], [216, -60]]]}]
        la = np.arange(-72.5, -59.5, 1)
        lo = np.arange(217, 291, 1)
#
    regional = var.rio.clip(geo, var.rio.crs)
    return regional, la, lo


# # # # Clipper map plot

# import cm_xml_to_matplotlib as cm
# import matplotlib.path as mpath
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import cartopy.crs as ccrs

# # file = '/Volumes/snati/data/pmi4-cmip6/pr_Amon_CAS-ESM2-0_historical_r1i1p1f1_gn_185001-201412.nc'
# # ds = xr.open_dataset(file, drop_variables='time_bnds')
# # ppt = ds['pr']

# loc = ['Austral', 'EAIS', 'W-WAIS/Ross', 'E-WAIS/Amundsen', 'Weddell Sea']

# # # aus, laa, loa = clipper(ppt.pr, loc[0])
# # # aus = aus.mean(['time', 'realization'])

# eais, lae, loe = clipper(dts['Rean'].pr, loc[1])
# eais = eais.mean(['time', 'realization'])

# ros, lar, lor = clipper(dts['Rean'].pr, loc[2])
# ros = ros.mean(['time', 'realization'])

# wais, law, low = clipper(dts['Rean'].pr, loc[3])
# wais = wais.mean(['time', 'realization'])

# wed, lad, lod = clipper(dts['Rean'].pr, loc[4])
# wed = wed.mean(['time', 'realization'])

# # lona, lata = np.meshgrid(loa, laa)
# lone, late = np.meshgrid(loe, lae)
# lonr, latr = np.meshgrid(lor, lar)
# lond, latd = np.meshgrid(lod, lad)
# lonw, latw = np.meshgrid(low, law)

# lev = np.linspace(0, 5, 5)
# # cmove = cm.make_cmap('ColorMoves/other-outl-1.xml', 'flip')
# # cbtick = np.linspace(0, 4, 11)

# cmove = plt.cm.Set3
# cax = plt.axes([0.1, 0.1, 0.8, 0.1])
# theta = np.linspace(0, 2 * np.pi, 100)
# center, radius = [0.5, 0.5], 0.6  # 0.5 = 90-50 = 40°
# verts = np.vstack([np.sin(theta), np.cos(theta)]).T
# circle = mpath.Path(verts * radius + center)
# map_proj = ccrs.Orthographic(0, -90.0)
# map_transf = ccrs.PlateCarree()


# plt.figure(figsize=[8, 8])
# ax = plt.axes(projection=map_proj)
# ax.set_boundary(circle, transform=ax.transAxes)
# ax.coastlines(zorder=10, color='gray')

# plt.contourf(lonw, latw, wais.where(wais * 0 != 0, 1), lev, cmap=cmove, transform=map_transf)

# plt.contourf(lone, late, eais.where(eais * 0 != 0, 2), lev, cmap=cmove, transform=map_transf)

# plt.contourf(lonr, latr, ros.where(ros * 0 != 0, 3), lev, cmap=cmove, transform=map_transf)

# plt.contourf(lond, latd, wed.where(wed * 0 != 0, 4), lev, cmap=cmove, transform=map_transf)

# gl = ax.gridlines(linestyle='--', zorder=3, color='lightgray')
# gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
# gl.ylocator = mticker.FixedLocator([-90, -85, -75, -65])
# plt.text(0, -85, '-85°', color='gray', horizontalalignment='right',
#          fontsize=10, transform=map_transf)
# plt.text(0, -75, '-75°', color='gray', horizontalalignment='right',
#          fontsize=10, transform=map_transf)
# plt.text(0, -65, '-65°', color='gray', horizontalalignment='right',
#          fontsize=10, transform=map_transf)
# # plt.subplots_adjust(left=0.3, bottom=0.3, right=0.6, top=0.5)
# plt.show()
