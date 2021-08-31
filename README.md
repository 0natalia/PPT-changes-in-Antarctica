# Precipitation changes in Antarctica
Set of Python codes that read reanalyses and CMIP6 models to addresses PPT and SST changes.

All scripts here read CMIP6 and reanalyses datasets as an ensemble and calculates their ens. mean.


* ### MSC_Taylor_Diagram.py
Estimates the secondary statistics necessary to plot the Taylor Diagram - RMS, Pearson correlation coefficient, and Standard Deviation.

Plots the Taylor Diagram

We used the python library Skill Metrics, available on
<https://github.com/PeterRochford/SkillMetrics.git>


* ### MSC_hist-amip_map.py
Calculates the difference in precipitation between the experiments (AMIP - CMIP = possible ocean influence on ppt)

Estimates the difference for the mean period 1979-2014, common for both experiments.

At last, this script plots: (i) map of the difference between the datasets

The map is overlaped with the multi-model agreement. The model agreement is defined by the ensemble size through binomial distribution. For achieving 95%  confidence level on agreement, in this study, 6/7 reanalyses and 15/23 models must agree.


* ### MSC_indices.py
Using the functions zw3_index and sam_index available on mscbib, it is acquires the 
Southern Annular Mode and the Zonal Wave 3 indices, plus secondary statistics regarding their
temporal evolution (trend, error, significance).

Plots: SAM/ZW3 index time series.


* ### MSC_mean-state-map.py
Addresses the mean PPT field south of 45• and the difference of each dataset from the ens. mean;

The map is overlaped with the multi-model agreement. The model agreement is defined by the ensemble size through binomial distribution. For achieving 95%  confidence level on agreement, in this study, 6/7 reanalyses and 15/23 models must agree.

At last, this script plots: (i) map of all-time average ppt for each dataset
                            (ii) map of the 50yr difference for each dset
                            

* ### MSC_seasonal_cycle.py
For the Antarctic regions defined in mscbib [Austral, EAIS, Ross, Weddell, and WAIS], the monthly data is grouped by month in order to
address the annual cycle.

This script plots: (i) ensemble mean annual cycle line plot
                   (ii) bar plot annual cycle for each dataset


* ### MSC_sst_vs_PPT_corr.py
This cript plots maps of the correlation (rPearson) between PPT regional time series (Antarctic regions according to mscbib) and SST field throughout 1900-2014.

* ### MSC_sst_vs_PPT_trendMap.py
For the Antarctic regions defined in mscbib [Austral, EAIS, Ross, Wed, and WAIS], the script estimates the variables' linear trend and other statistics.

This script plots: PPT trend and SST trend.


* ### MSC_sst_vs_PPT_regress.py
For the Antarctic regions defined in mscbib [Austral, EAIS, Ross, Wed and WAIS], it estimates the variables linear trend and secondary statistics.

This script plots: PPT trend X SST trend, and their correlation as R2.


* ### MSC_timeseries_and_trends.py
Estimates the precipitation linear trend for Antarctic regions [Austral, EAIS, Ross, Wed, and WAIS].

*Cut regions using clipper function in mscbib*

This script plots: (i) regional precipitation time series
                            (ii) regional ppt anomalies relative to 1979:2010
                            (iii) calculated trends for each dataset
                                  as barplot
                            (iv) linear fit time series
                            (v) ppt trend per dataset as a heatmap



Note: always calculate reanalysis-ensemble mean and model-ensemble
mean separately. Do not mix different data types.
