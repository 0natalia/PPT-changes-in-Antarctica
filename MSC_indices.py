# -*- coding: utf-8 -*-
'''
PPT changes in high southern latitudes

This script reads renalyses and calculates its ensamble mean.

Using the functions zw3_index and sam_index available on mscbib,
it is acquired the indices along with secondary statistics of their
temporal evolution (trend, error, significance)

Natália Silva (2021)
natalia3.silva@usp.br
'''

from mscbib import my_cmap, tinv, ptext, sam_index, zw3_index
import matplotlib.pyplot as plt
import numpy as np
import glob
import xarray as xr
import datetime
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")

plt.style.use('seaborn-darkgrid')
plt.rcParams["figure.figsize"] = (14, 5)
sns.set_context('talk')  # large fontsize
c = my_cmap(8, 'R')


# *_*_*_*_*_*_*_*_*_*_*   READ AND CONCATENATE DATA   _*_*_*_*_*_*_*_*_*_*
# *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_

# ####### Reanalysis #######
files = sorted(glob.glob('/Volumes/snati/data/SLP/*.nc'))
dset = []
for f in files:
    ds = xr.open_dataset(f, drop_variables=[
                         'initial_time0_encoded', 'time_bnds'])
    ds.sel(lat=slice(-89.5, -40.5))
    dset.append(ds)


# Change calendars for monthly data
for i in range(len(dset)):
    if i == 1:
        ti = datetime.date(1979, 1, 1)
        tf = datetime.date(2011, 1, 1)
        dset[i]['time'] = np.arange(ti, tf, dtype='datetime64[M]')
    elif i == 5:
        ti, tf = dset[i].time[0], dset[i].time[-3].values + \
            np.timedelta64(31, 'D')
        dset[i]['time'] = np.arange(ti.values, tf, dtype='datetime64[M]')
    else:
        ti, tf = dset[i].time[0], dset[i].time[-1].values + \
            np.timedelta64(31, 'D')
        dset[i]['time'] = np.arange(ti.values, tf, dtype='datetime64[M]')


rean = ['20thCV3', 'CFSR', 'ERA-20', 'ERA-I', 'JRA55', 'MERRA2', 'NCEP2']
data = xr.concat(dset, dim='realization').assign_coords({'realization': rean})
p = data.sel(time=slice('1900', '2014'))
p['slp'] /= 100
slp = xr.concat([p, p.mean('realization').assign_coords(
    {'realization': 'MRM7'})], dim='realization')


# *_*_*_*_*_*_*_* Calculate the Southern Annular Mode Index *_*_*_*_*_*_*_*_*_
idx = sam_index(slp)
# idx = zw3_index(slp)

tic = np.arange('1900-01', '2015-01', 120, dtype='datetime64[M]')
lab = ['1900', '1910', '1920', '1930', '1940', '1950',
       '1960', '1970', '1980', '1990', '2000', '2010']

ax = idx['idx'].loc[:, :'NCEP2'].plot(color=c, linewidth=1, alpha=1,
                                      zorder=1, xticks=[])
idx['idx']['MRM7'].plot(linewidth=2, color='black', alpha=0.4, xticks=tic)
idx['linebef'].plot(color='black')
idx['lineaft'].plot(color='black')
plt.xlabel('yr')
plt.ylabel('I$_{SAM}$')
lim = ax.get_ylim()
plt.ylim(lim)
plt.xlim(idx['idx'].index[0], idx['idx'].index[-1])
plt.xticks(tic, lab)
plt.text(idx['idx'].index[5], lim[1] - 1, 'BEF80: ' + f"{idx['fitBEF'].slope * 10: .2e} +/- {idx['errob'] * idx['fitBEF'].stderr * 10: .2e}" + '; ' + idx['pB80'])
plt.text(idx['idx'].index[5], lim[1] - 3, 'AFT80: ' + f"{idx['fitAFT'].slope * 10: .2e} +/- {idx['errob'] * idx['fitAFT'].stderr * 10: .2e}" + '; ' + idx['pA80'])
plt.subplots_adjust(left=0.07, bottom=0.15, right=0.8)
ax.legend(bbox_to_anchor=(0.25, 0.1, 0.95, 0.95))
plt.show()
# plt.savefig('zw3.pdf', dpi=50, facecolor='w', format='pdf')
plt.close()


# FIM
