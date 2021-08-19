for i in psl*Can*.nc;
do
echo $i

# # Correct ERA5 and Era-Interim total cummulative ppt
# cdo -b F64 daysum -shifttime,-1hour "$i" "pr_EI_${i}.nc"

# Rename variables
# ncrename -d initial_time0_hours,time -v initial_time0_hours,time -v PRATE_P8_L1_GGA0_avg,pr inf3.nc inf4.nc

# # CDO commands

# cdo select,name=TMP_P8_L1_GLL0 "$i" "${i}_Z.nc"

cdo -remapbil,r360x180 "$i" inf1.nc

cdo -sellonlatbox,0,360,-90,-30 inf1.nc inf2.nc

cdo -b F64 -divc,100 inf2.nc inf3.nc

cdo -setunit,'hPa' inf3.nc inf4.nc

# cdo -yearmean inf2.nc "${i}_Z.nc"

# Remove variables
ncks -C -O -x -v time_bnds inf4.nc inf5.nc
ncks -C -O -x -v lat_bnds inf5.nc inf6.nc
ncks -C -O -x -v lon_bnds inf6.nc "${i}_F.nc"

# # # Remove dimension
# ncwa -a bnds inf7.nc "${i}_F.nc"

rm inf*.nc 

done