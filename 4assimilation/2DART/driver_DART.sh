#!/bin/sh
#BSUB -J DART_FKC
#BSUB -q fat_384
#BSUB -n 80
#BSUB -oo test.out
#==========================
#assim time               
day=10                    
hour=00                   
minute=00                 
#==========================
#assim parameters
# cutoff=0.1                                            #horizontal localization=2*cutoff*6400(km)
# vert_normalization_height=800000.0                      #vertical localization = 2*cutoff*vert_normalization_height(meters)
flag_single_obs=0
cutoff_list=(0.05)
vert_norm_list=(80000.0) #(80000.0 400000.0 800000.0)
#========================
if [ "$flag_single_obs" -eq 1 ]; then
	echo "singleobs"
	suffix=singleobs
	obs_file=obs_seq.out_kctest1_d01_${day}${hour}_${suffix}
else
	suffix=allobs
	obs_file=obs_seq.out_kctest1_d01_${day}_${hour}_qunatile
fi
for cutoff in "${cutoff_list[@]}"; do
	for vert_normalization_height in "${vert_norm_list[@]}"; do

		#===========================
		horiz_result=$(echo "$cutoff * 6400 * 2 " | bc)
		horizLoc=$(printf "%.0f" "$horiz_result")   #the horizontal localization in km
		vert_result=$(echo "$cutoff * $vert_normalization_height * 2 / 1000" | bc)
		vertLoc=$(printf "%.0f" "$vert_result")   #the vertical localization in km
		echo $horizLoc
		echo $vertLoc 

		#==========================
                main_dir=/share/home/lililei1/kcfu/tc_mangkhut
		obs_dir=/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/1convert_obs/run_dir
		work_dir=/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/2DART
		DART_DIR=/share/home/lililei1/kcfu/models/DART_main/models/wrf/work
		ensmem_dir=/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/${day}_${hour}_${minute}
		mkdir -p ${work_dir}/run_dir ${work_dir}/postAssimData/${day}_${hour}_${minute}/${suffix}
		#==========================
		#link necessary files to run_dir
		cd ${work_dir}/run_dir
		if [ -e ${work_dir}/run_dir/output_mean.nc ];then
			rm output_mean.nc
		fi
		ln -sf ${DART_DIR}/filter .
		ln -sf ${DART_DIR}/rtcoef_noaa_18_amsua.dat .
		ln -sf ${DART_DIR}/rttov_sensor_db.csv .
		ln -sf ${DART_DIR}/output_list_d01.txt .
		ln -sf ${DART_DIR}/ncks.sh
		cp ${main_dir}/0necessay_files/input.nml .
		ln -sf ${ensmem_dir}/input_list_d01.txt .
		ln -sf ${ensmem_dir}/firstguess.ensmean ./wrfinput_d01

		ln -sf ${obs_dir}/${obs_file} ./obs_seq.out
        #ln -sf ${obs_dir}/obs_seq.out_kctest1_d01_1000_singleobs ./obs_seq.out
		sed -i "s/cutoff[[:space:]]*=.*/cutoff                        =   ${cutoff},/g" input.nml
		sed -i "s/vert_normalization_height[[:space:]]*=.*/vert_normalization_height   =  ${vert_normalization_height},/g" input.nml

		#==========================
                rm dart_log.out
		mpirun ./filter
		rm *errors*
		cp wrfinput_d01 analysis.ensmean
		./ncks.sh
	    mv analysis.ensmean ${work_dir}/postAssimData/${day}_${hour}_${minute}/${suffix}/analysis.ensmean.horizLoc${horizLoc}.vertLoc${vertLoc}
		for ((imem=1;imem<=50;imem++));do
			member="mem"`printf %03i ${imem}`
			mv output_${member} ${work_dir}/postAssimData/${day}_${hour}_${minute}/${suffix}/analysis.${member}.horizLoc${horizLoc}.vertLoc${vertLoc}
		done
		cp dart_log.out ${work_dir}/postAssimData/${day}_${hour}_${minute}/${suffix}/dart_log.out.horizLoc${horizLoc}.vertLoc${vertLoc}
	done
done
