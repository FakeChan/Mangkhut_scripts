#!/bin/sh

#BSUB -n 1
#BSUB -J fkcrttovens
#BSUB -oo hx_rttov_ens_fkc.log
#BSUB -eo hx_rttov_ens_fkc.err
#BSUB -q serial

#==============================
#set instrument
export python_bin=/share/home/lililei1/kcfu/anaconda/envs/wrf/bin
export instrument=AMSUA
if [[ "$instrument" == "AMSUA" ]];then
	export satlon=144.7
	export satheight=854
elif [[ "$instrument" == "GIIRS" ]]; then
	export satlon=104.7
	export satheight=35793
fi
#==============================
#set domain

export domain=d01

export npoint=676
export nlevels=56

export i_parent_start=27 #same as namelist, used in matlab code to get profile
export j_parent_start=88
#export rttov_scatt=0 #${rttov_scatt:0} # 0: simple cloud scheme, 1: rttov_scatt
export use_total_ice=0 #0: seperately calculate scatt effect of Snow and Ice; 1: use total ice
#==============================
#set obs time
if [ ! -z "${cycle_flag}" ]; then  #this script is used in cycling DA
  export obs_day=$current_day
  export obs_hour=$current_hour
  export obs_min=$current_min
else  #this script is used in non-cycling DA
  export obs_day=10
  export obs_hour=00
  export obs_min=00
fi
#==============================
#set work dir
export work_dir=/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov 
export run_matlab_dir=${work_dir}/profile
# export ens_wrfout_dir=/share/home/lililei1/kcfu/tc_mangkhut/2ens_free_fcst/
export ens_wrfout_dir=/scratch/lililei1/kcfu/tc_mangkhut/2ens_free_fcst/
export rttov_dir=/share/home/lililei1/kcfu/models/rttov123
export prof_dir=${work_dir}/profile/profile_${domain}
export obs_dir=${work_dir}/4ens_BT
export merge_dir=${obs_dir}/merge
export add_pert_dir=${obs_dir}/add_pert

mkdir -p ${prof_dir} ${obs_dir} ${merge_dir} ${add_pert_dir}
memlist=({1..50})
#memlist=(1)
for imem in ${memlist[*]};do
	export member="mem"`printf %03i ${imem}`
	if [[ ! -d ${ens_wrfout_dir}/${member} ]];then
		continue
	fi
	echo "$member"
	export ens_mem_dir=${ens_wrfout_dir}/${member}/
	export prof_mem_dir=${prof_dir}/${member}/
	export obs_mem_dir=${obs_dir}/${member}/
	mkdir -p ${ens_mem_dir} ${prof_mem_dir} ${obs_mem_dir}
	#==============================
	#step 1 run matlb to generate profile

	cd ${run_matlab_dir}/2ens_matlab
	./${domain}_run_matlab.sh
	echo "profile done"

	#===============================
	#step 2 call rttov

	if [[ "$instrument" == "AMSUA" ]];then
		export rtcoef_dir=${rttov_dir}/rtcoef_rttov12/rttov7pred54L
		export rtcoef=rtcoef_noaa_18_amsua.dat
		export chnum=6
		
	elif [[ "$instrument" == "MHS" ]]; then
		export rtcoef_dir=${rttov_dir}/rtcoef_rttov11/rttov7pred54L
		export rtcoef=rtcoef_noaa_18_mhs.dat
	elif [[ "$instrument" == "GIIRS" ]]; then
		export rtcoef_dir=${rttov_dir}/rtcoef_rttov11/rttov7pred101L
		export rtcoef=rtcoef_fy4_1_giirs_local.dat
		export chnum=1650
	fi
	ln -sf ${rtcoef_dir}/${rtcoef} ${work_dir}/2call_rttov/

	bash ${work_dir}/2call_rttov/${domain}_call_rttov.sh
	echo "rttov done"

	#==============================
	#step 3 seperate obs into different channels

	bash ${merge_dir}/5_hebing_diffchan_d01.sh

done
