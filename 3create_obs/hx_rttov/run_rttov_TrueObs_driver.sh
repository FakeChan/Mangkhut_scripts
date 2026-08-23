#!/bin/sh

#BSUB -n 1
#BSUB -J fkcrttov1
#BSUB -oo hx_rttov_fkc.log
#BSUB -eo hx_rttov_fkc.err
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
elif [[ "$instrument" == "AMSR2" ]]; then
        export satlon=144.7
        export satheight=700
fi
#==============================
#set domain

export domain=d01

export npoint=676
export nlevels=56
export obserr_std=${obs_err_std:-0.5}
export rttov_scatt="${rttov_scatt:-0}" # 0 simple cloud scheme; 1 RTTOV-SCATT
export use_total_ice="${use_total_ice:-0}" #0: seperately calculate scatt effect of Snow and Ice; 1: use total ice
export lacc_mode=0
#==============================
#set obs time (priority: obs_* > current_* > defaults)
export obs_day="${obs_day:-${current_day:-10}}"
export obs_hour="${obs_hour:-${current_hour:-00}}"
export obs_min="${obs_min:-${current_min:-00}}"

echo "truth rttov time: ${obs_day}_${obs_hour}_${obs_min}"
#==============================
#set work dir
export work_dir=/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov 
export run_matlab_dir=${work_dir}/profile
export NR_wrfout_dir=/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain/
export rttov_dir=/share/home/lililei1/kcfu/models/rttov123
export prof_dir=${work_dir}/profile/profile_${domain}
export obs_dir=${work_dir}/3obs_BT
export merge_dir=${obs_dir}/merge
export add_pert_dir=${obs_dir}/add_pert
#==============================
#step 1 run matlb to generate profile
mkdir -p ${prof_dir} ${obs_dir} ${merge_dir} ${add_pert_dir}
cd ${run_matlab_dir}/1run_matlab
./${domain}_run_matlab.sh
echo "profile done"

#===============================
#step 2 call rttov

if [[ "$instrument" == "AMSUA" ]];then
	export rtcoef_dir=${rttov_dir}/rtcoef_rttov12/rttov7pred54L
	export rtcoef=rtcoef_noaa_18_amsua.dat
	export chnum=15
        export MIETABLE_DIR=${rttov_dir}/rtcoef_rttov12/mietable
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

bash ${work_dir}/2call_rttov/${domain}_NR_call_rttov.sh
echo "rttov done"

#==============================
#step 3 seperate obs into different channels

bash ${merge_dir}/5_hebing_diffchan_d01.sh

#==============================
#step 4 add pert
bash ${add_pert_dir}/add_pert.sh

#==============================
#step 5 place the clear-sky mask in the obs BT dir
#  matlab writes it to ${prof_dir}/clear_sky_mask_${obs_day}_${obs_hour}:${obs_min}.txt
#  (time string contains ':'), while the single-test pipeline expects
#  ${obs_dir}/${instrument}/BT_${obs_day}_${obs_hour}_${obs_min}/clear_sky_mask.txt.
#==============================
if [[ "${rttov_scatt}" == "0" ]]; then
    mask_src="${prof_dir}/clear_sky_mask_${obs_day}_${obs_hour}:${obs_min}.txt"
    mask_dst_dir="${obs_dir}/${instrument}/BT_${obs_day}_${obs_hour}_${obs_min}"
    if [[ ! -f "${mask_src}" ]]; then
        echo "WARNING: clear-sky mask not found: ${mask_src}" >&2
    else
        mkdir -p "${mask_dst_dir}"
        cp -f "${mask_src}" "${mask_dst_dir}/clear_sky_mask.txt"
        echo "clear-sky mask placed at: ${mask_dst_dir}/clear_sky_mask.txt"
    fi
fi
