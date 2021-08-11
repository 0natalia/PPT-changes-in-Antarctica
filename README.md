# Precipitation_changes_during_the_Hist_Period_Antarctica
Set of Python codes that read reanalyses and CMIP6 models to addresses PPT and SST changes.

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
