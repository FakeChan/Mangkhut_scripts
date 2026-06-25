#!/bin/bash

#BSUB -n 1
#BSUB -J fkcrttov_lacc
#BSUB -oo hx_rttov_LACC_fkc.log
#BSUB -eo hx_rttov_LACC_fkc.err
#BSUB -q serial

set -e

#==============================
# set instrument
export python_bin=/share/home/lililei1/kcfu/anaconda/envs/wrf/bin
export instrument=AMSUA
if [[ "$instrument" == "AMSUA" ]];then
	export satlon=144.7
	export satheight=854
elif [[ "$instrument" == "GIIRS" ]]; then
	export satlon=104.7
	export satheight=35793
elif [[ "$instrument" == "AMSR2" ]]; then
	export satlon=144.7
	export satheight=700
fi

#==============================
# set domain and LACC window
export domain=d01
export npoint=676
export nlevels=56
export obserr_std=${obs_err_std:-0.5}
export rttov_scatt=${rttov_scatt:-0}
export use_total_ice=0
export lacc_mode=1

if [ ! -z "${cycle_flag}" ]; then
	export lacc_center_day=${lacc_center_day:-$current_day}
	export lacc_center_hour=${lacc_center_hour:-$current_hour}
	export lacc_center_min=${lacc_center_min:-$current_min}
else
	export lacc_center_day=${lacc_center_day:-10}
	export lacc_center_hour=${lacc_center_hour:-00}
	export lacc_center_min=${lacc_center_min:-00}
fi
export lacc_center_time=${lacc_center_day}_${lacc_center_hour}_${lacc_center_min}

build_lacc_times() {
	local center_day_num=$((10#${lacc_center_day}))
	local center_hour_num=$((10#${lacc_center_hour}))
	local center_min_num=$((10#${lacc_center_min}))
	local times=""
	local lag lag_day lag_hour

	for lag in ${LACC_LAG_HOURS:-3 6 9};do
		lag_day=${center_day_num}
		lag_hour=$((center_hour_num - lag))
		while (( lag_hour < 0 ));do
			lag_hour=$((lag_hour + 24))
			lag_day=$((lag_day - 1))
		done
		times="${times} $(printf "%02d_%02d_%02d" ${lag_day} ${lag_hour} ${center_min_num})"
	done
	echo "${times# }"
}

# Override with LACC_TIMES="09_21_00 09_18_00 09_15_00"; otherwise use LACC_LAG_HOURS.
if [[ ! -z "${LACC_TIMES}" ]];then
	export lacc_times="${LACC_TIMES}"
else
	export lacc_times="$(build_lacc_times)"
fi
echo "LACC truth center time: ${lacc_center_time}"
echo "LACC truth lag times: ${lacc_times}"

#==============================
# set work dir
export work_dir=/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov
export run_matlab_dir=${work_dir}/profile
export NR_wrfout_dir=/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain/
export rttov_dir=/share/home/lililei1/kcfu/models/rttov123
export prof_dir=${work_dir}/profile/profile_${domain}_LACC_${lacc_center_time}
export obs_dir=${work_dir}/3obs_BT_LACC
export merge_dir=${work_dir}/3obs_BT/merge
export add_pert_dir=${obs_dir}/add_pert
mkdir -p ${prof_dir} ${obs_dir} ${add_pert_dir}

#===============================
# set RTTOV coefficients
if [[ "$instrument" == "AMSUA" ]];then
	export rtcoef_dir=${rttov_dir}/rtcoef_rttov12/rttov7pred54L
	export rtcoef=rtcoef_noaa_18_amsua.dat
	export chnum=15
	export MIETABLE_DIR=${rttov_dir}/rtcoef_rttov12/mietable
	export MIETABLE=mietable_noaa_amsua.dat
elif [[ "$instrument" == "MHS" ]]; then
	export rtcoef_dir=${rttov_dir}/rtcoef_rttov12/rttov7pred54L
	export rtcoef=rtcoef_noaa_18_mhs.dat
elif [[ "$instrument" == "GIIRS" ]]; then
	export rtcoef_dir=${rttov_dir}/rtcoef_rttov12/rttov7pred101L
	export rtcoef=rtcoef_fy4_1_giirs_local.dat
	export chnum=1650
elif [[ "$instrument" == "AMSR2" ]]; then
	export rtcoef_dir=${rttov_dir}/rtcoef_rttov12/rttov7pred54L
	export rtcoef=rtcoef_gcom-w_1_amsr2.dat
	export chnum=7
	export MIETABLE_DIR=${rttov_dir}/rtcoef_rttov12/mietable
	export MIETABLE=mietable_gcom-w_amsr2.dat
fi
ln -sf ${rtcoef_dir}/${rtcoef} ${work_dir}/2call_rttov/

for lacc_time in ${lacc_times};do
	export obs_day=${lacc_time%%_*}
	_lacc_hm=${lacc_time#*_}
	export obs_hour=${_lacc_hm%%_*}
	export obs_min=${_lacc_hm##*_}
	export time=${obs_day}_${obs_hour}_${obs_min}

	echo "======================================================"
	echo "LACC truth lag time: ${time}; center fixed at ${lacc_center_time}"
	echo "======================================================"

	cd ${run_matlab_dir}/1run_matlab
	./${domain}_run_matlab.sh
	echo "profile done"

	bash ${work_dir}/2call_rttov/${domain}_NR_call_rttov.sh
	echo "rttov done"

	mkdir -p ${obs_dir}/${instrument}/BT_${time}
	cd ${merge_dir}
	matlab -nodesktop -nosplash -nodisplay < hebing_diffchan.m
	echo "merge done"
done

cd ${merge_dir}
matlab -nodesktop -nosplash -nodisplay < average_LACC_obs.m
echo "LACC obs average done: ${obs_dir}/${instrument}/BT_LACC_${lacc_center_time}"
