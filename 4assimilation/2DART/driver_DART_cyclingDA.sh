#!/bin/sh
#BSUB -J DART_FKC
#BSUB -q largemem
#BSUB -n 120
#BSUB -R "span[ptile=24]"
#BSUB -oo DART.out
#BSUB -eo DART.err
#==========================
#assim time               
day=${current_day}                    
hour=${current_hour}                   
minute=${current_min}                 
#==========================
#assim parameters
# cutoff=0.1                                            #horizontal localization=2*cutoff*6400(km)
# vert_normalization_height=800000.0                      #vertical localization = 2*cutoff*vert_normalization_height(meters)
flag_single_obs=0
flag_filter=${filter_kind}
cutoff_list=(0.05)
vert_norm_list=(80000.0) #(80000.0 400000.0 800000.0)
#========================
rm ${work_dir}/run_dir/DART_done_cycling_${cycle_count}  #clear previous file
max_dom=${max_dom:-2}       #if max_dom not in envs, set max_dom=2
if [ "$flag_single_obs" -eq 1 ]; then
	echo "singleobs"
	suffix=singleobs
	obs_file=obs_seq.out_kctest1_d01_${day}${hour}_${suffix}
else
	suffix=allobs
	obs_file=${obs_seq_name}
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
        main_dir=${base_dir}
		obs_dir=${main_dir}/4assimilation/1convert_obs/run_dir
		work_dir=${main_dir}/4assimilation/2DART
		DART_DIR=${DART_DIR}
		
		
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
		ln -sf ${DART_DIR}/ncks.sh .
		cp ${main_dir}/0necessay_files/input.nml .
		
		if [ "$profile_matlab_flag" -eq 1 ]; then
			#if this flag=1 ,it denotes that this is the initial DA at 10_00_00
			export ensmem_dir=/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/cyclingDA/${current_time}
			echo "fg dir: ${ensmem_dir}"
			#note: here we do not link initial wrfout into ensmemdir,as later in gen_wrfout_list.py this will be done.
			${python_bin}/python ${main_dir}/4assimilation/gen_input_list_cycling.py
			ln -sf ${ensmem_dir}/input_list_d0*.txt .
			${python_bin}/python ${main_dir}/4assimilation/gen_output_list.py  # if this is the first DA time, create  new ouput_list files

			#now input_list output_list are in xxx/2DART/run_dir 
		else
			#if not, then call python to generate
			export ensmem_dir=${cycle_wrfout_dir}
			${python_bin}/python ${main_dir}/4assimilation/gen_input_list_cycling.py 
			ln -sf ${ensmem_dir}/input_list_d0*.txt .
		fi 
		    #warning : input_list_d01 is created manually.
		cd ${work_dir}/run_dir
		for ((dom = 1; dom <= ${max_dom}; dom += 1)); do
			ln -sf ${ensmem_dir}/firstguess_d0${dom}.mem001 ./wrfinput_d0${dom} # DART need 
		done 

		ln -sf ${obs_dir}/${obs_file} ./obs_seq.out
        #ln -sf ${obs_dir}/obs_seq.out_kctest1_d01_1000_singleobs ./obs_seq.out

		#=================================modify input.nml for DART==================================
		sed -i "s/cutoff[[:space:]]*=.*/cutoff                        =   ${cutoff},/g" input.nml
		sed -i "s/vert_normalization_height[[:space:]]*=.*/vert_normalization_height   =  ${vert_normalization_height},/g" input.nml
		sed -i "s/num_domains[[:space:]]*=.*/num_domains   =  ${max_dom},/g" input.nml

		input_list_str=""
		output_list_str=""
		for ((dom = 1; dom <= ${max_dom}; dom += 1)); do
			dom_id=$(printf "%02d" $dom)
			# cat str
			if [ "$dom" -eq 1 ]; then
				# first domain
				input_list_str="\"input_list_d${dom_id}.txt\""
				output_list_str="\"output_list_d${dom_id}.txt\""
			else
				# domain sequence
				input_list_str="${input_list_str}, \"input_list_d${dom_id}.txt\""
				output_list_str="${output_list_str}, \"output_list_d${dom_id}.txt\""
			fi
		done 
		sed -i "s|.*input_state_file_list.*=.*|   input_state_file_list = ${input_list_str}|" input.nml
		sed -i "s|.*output_state_file_list.*=.*|   output_state_file_list = ${output_list_str}|" input.nml
        #============================================================================================
		if [ "$flag_filter" -eq 1 ]; then
			echo "using EAKF"
		elif [ "$flag_filter" -eq 2 ];then
		   sed -i "s/qceff_table_filename[[:space:]]*=.*/qceff_table_filename  =   'qceff_table_fkc.csv',/g" input.nml
		   echo "using QCF_RHF"
		fi
	

		#====================bias correction at the first DA time on OM_TMP======================
		if [ "$profile_matlab_flag" -eq 1 ]; then
			cp ${main_dir}/0necessay_files/correct_bias.py .
			cp ${main_dir}/4assimilation/gen_wrfout_list.py .
			${python_bin}/python gen_wrfout_list.py
			#note here in gen_wrfout_list.py, initial wrfout is linked into ensmem_dir
			${python_bin}/python correct_bias.py
		fi
		#========================================================================================

        rm dart_log.out
		mpirun ./filter 
		rm *errors*

		

		#===============================update firstguess file to analysis file==========================
		cp ${main_dir}/0necessay_files/ncks_updateDARTvar.sh ./ncks.sh
                # first remove all analysis file in post_anal_dir
                rm ${post_anal_dir}/analysis_d01.mem*
                rm ${post_anal_dir}/analysis_d02.mem*
		for ((dom = 1; dom <= ${max_dom}; dom += 1)); do
			for ((imem=1;imem<=50;imem++));do
				member="mem"`printf %03i ${imem}`
				cp ${ensmem_dir}/firstguess_d0${dom}.${member} ./firstguess_d0${dom}.${member}
				cp ./output_d0${dom}.${member} ./analysis_d0${dom}.${member}
				export base_file=firstguess_d0${dom}.${member}  #base_file is the file to be updated (fg)
				export updated_file=analysis_d0${dom}.${member}   #updated file is the file already updated after DART
				./ncks.sh
				mv firstguess_d0${dom}.${member} ${post_anal_dir}/analysis_d0${dom}.${member}
			done
			cp dart_log.out ${post_anal_dir}/dart_log.out
			ncea -O ${post_anal_dir}/analysis_d0${dom}* ${post_anal_dir}/analysis_d0${dom}.ensmean
                        	
		done
                rm analysis_d0*
                rm output_d0*
		#===================================================================================================
	done
done

touch ${work_dir}/run_dir/DART_done_cycling_${cycle_count}
