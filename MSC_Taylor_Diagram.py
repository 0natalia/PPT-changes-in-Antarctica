# -*- coding: utf-8 -*-
'''
This script reads sst and ppt data from Reanalyses, and CMIP6,
and calculates their ens. mean.
Then, it estimates the secondary statistics necessary to plot the Taylor
Diagram - RMS, Pearson correlation coefficient, and Standard Deviation.
Finally, the script plots the Taylor Diagram, in order to provide a way of
graphically summarizing how closely a pattern (simulations) matches
observations (Karl E. Taylor, 2001).

We used the python library Skill Metrics, available on
<https://github.com/PeterRochford/SkillMetrics.git>
Courtesy of Laura Sobral Verona.

Natália Silva (2021)
natalia3.silva@usp.br
'''

import matplotlib.pyplot as plt
import numpy as np
import skill_metrics as sm
from collections import defaultdict
import xarray as xr
import glob
import seaborn as sns


plt.rcParams["figure.figsize"] = (7, 5)
sns.set_context('talk', font_scale=1)  # large fontsize


# *_*_*_*_*_*_*_*_*_*_*   READ AND CONCATENATE DATA   _*_*_*_*_*_*_*_*_*_*
# *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_

# ####### CMIP6 pr + tos #######
path = ['/Volumes/snati/data/PPT/cmip6_hist/hist/*.nc',
        '/Volumes/snati/data/SST/cmip6_hist/*.nc']
m = {'pr': [], 'tos': []}

for p, n in zip(path, m):
    files = sorted(glob.glob(p))
    dset, name = [], []
    for f in files:
        ds = xr.open_dataset(f, drop_variables=[
                             'time_bnds', 'initial_time0_encoded'])
        ds = ds.assign_coords({'realization': ds.source_id})
        ds = ds.sel(lat=slice(-89.5, -40.5), time=slice('1900', '2014'))
        ds['time'] = ds.time.dt.year
        dset.append(ds)
        name.append(ds.source_id)
        #
        #
    m[n] = xr.concat(dset, 'realization')
    m[n] = xr.concat([m[n], m[n].mean('realization').assign_coords(
        {'realization': 'CMIP6'})], dim='realization')
    #


# ####### Reanalysis #######
path = ['/Volumes/snati/data/PPT/reanalise/pr*.nc',
        '/Volumes/snati/data/SST/obs-rean/sst*.nc']
rname = ['20thCv3', 'CFSR', 'Era20C', 'EraInt', 'JRA55', 'MERRA2', 'NCEP2']
rean = {'pr': [], 'tos': []}

for p, n in zip(path, rean):
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
    rean[n] = xr.concat(dset, 'realization').assign_coords({'realization': rname})
    rean[n] = xr.concat([rean[n], rean[n].mean('realization').assign_coords(
        {'realization': 'REAN'})], dim='realization')



# del(ds, dset, f, files, glob, obs, p, path)

# *_*_*_*_*_*_*_*_*_*_*_*  TAYLOR DIAGRAM PRIMER  *_*_*_*_*_*_*_*_*_*_*_*_
# *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_
# Calculate statistics for Taylor diagram
# The first array element corresponds to the reference series
# for the while the second is that for the predicted series.

# for n in d:
n = 'pr'

if n == 'pr':
    axmax = 0.031
    tcrms = [0.005, 0.015, 0.025]
    tcstd = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
elif n == 'tos':
    axmax = 0.22
    tcrms = [0.05, 0.15, 0.2]
    tcstd = [0.05, 0.1, 0.15, 0.2, 0.22]


# Recortar modelos p/ período comum com rean/obs
rf = rean[n]  # rean
ref = rf[n].mean(['lat', 'lon']).sel(realization='REAN').to_dict()

dic = defaultdict(list)

name = m[n].realization.values.tolist()
for j in name:
    pred = m[n].sel(realization=j)[n].mean(['lat', 'lon']).to_dict()
    ref_stats = sm.taylor_statistics(ref, pred, 'data')
    dic['ccoef'].append(ref_stats['ccoef'][1])  # r
    dic['crmsd'].append(ref_stats['crmsd'][1])  # RMS
    dic['sdev'].append(ref_stats['sdev'][0])  # std


dic['ccoef'].insert(1, dic['ccoef'][0])  # r
dic['crmsd'].insert(1, dic['crmsd'][0])  # RMS
dic['sdev'].insert(1, dic['sdev'][0])  # std
name.insert(0, 'a')
label = np.arange(0, len(name)).tolist()
label[-1] = 'CMIP6'

plt.close()
sm.taylor_diagram(np.array(dic['sdev']), np.array(dic['crmsd']),
                  np.array(dic['ccoef']),
                  markerLegend='off', axismax=axmax,
                  markerLabelColor='black', markerLabel=label,
                  markerSize=15, cmapzdata=np.array(dic['ccoef']),
                  checkstats='on', styleOBS='-', colOBS='gold',
                  markerobs='x', titleOBS='REAN',
                  tickRMSangle=115, showlabelsRMS='on',
                  titleRMS='on', tickRMS=tcrms, tickSTD=tcstd,
                  colRMS='greenyellow', styleRMS='-', widthRMS=1.0,
                  colSTD='black', styleSTD='-.', widthSTD=1.0, titleSTD='on',
                  colCOR='gray', styleCOR=':', widthCOR=1.0, titleCOR='on')
plt.show()


# FIM