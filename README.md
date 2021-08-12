# Precipitation changes in Antarctica
Set of Python codes that read reanalyses and CMIP6 models to addresses PPT and SST changes.

* ### MSC_timeseries_and_trends.py
Reads CMIP6 and reanalyses datasets as an ensemble and calculates their ens. mean.

Estimates the precipitation linear trend for Antarctic regions [Austral, EAIS, Ross, Wed, and WAIS].
*Cut regions using clipper function in mscbib

This script plots: (i) regional precipitation time series
                            (ii) regional ppt anomalies relative to 1979:2010
                            (iii) calculated trends for each dataset
                                  as barplot
                            (iv) linear fit time series
                            (v) ppt trend per dataset as a heatmap

* ### MSC_Taylor_Diagram.py
Reads sst and ppt data from Reanalyses, and CMIP6;

Estimates the secondary statistics necessary to plot the Taylor Diagram - RMS, Pearson correlation coefficient, and Standard Deviation.

Plots the Taylor Diagram

We used the python library Skill Metrics, available on
<https://github.com/PeterRochford/SkillMetrics.git>


* ### MSC_mean-state-map.py
Reads ppt data from Reanalyses, and CMIP6;

Addresses the mean PPT field south of 45• and the difference of each dataset from the ens. mean;

The map is overlaped with the multi-model agreement. The model agreement is defined by the ensemble size through binomial distribution. For achieving 95%  confidence level on agreement, in this study, 6/7 reanalyses and 15/23 models must agree.

At last, this script plots: (i) map of all-time average ppt for each dataset
                            (ii) map of the 50yr difference for each dset
                            

* ### MSC_seasonal_cycle.py
Reads CMIP6 (AMIP & Hist) and reanalysis datasets and calculates their ensemble means;

For the Antarctic regions defined in mscbib [Austral, EAIS, Ross, Weddell, and WAIS], the monthly data is grouped by month in order to
address the annual cycle.

This script plots: (i) ensemble mean annual cycle line plot
                   (ii) bar plot annual cycle for each dataset


Note: always calculate reanalysis-ensemble mean and model-ensemble
mean separately. Do not mix different data types.
