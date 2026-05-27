#!/bin/sh

#BSUB -n 1
#BSUB -J fkcrttov1
#BSUB -oo hx_rttov_fkc.log
#BSUB -eo hx_rttov_fkc.err
#BSUB -q serial

#==============================
#set instrument
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
export obserr_std=${obs_err_std:-0.5}
#export rttov_scatt=1 # 0 simple cloud scheme; 1 RTTOV-SCATT
export use_total_ice=1 #0: seperately calculate scatt effect of Snow and Ice; 1: use total ice
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

echo "truth rttov time: ${obs_day}_${obs_hour}_${obs_min}"
#==============================
#set work dir
export work_dir=/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov 
export run_matlab_dir=${work_dir}/profile
export NR_wrfout_dir=/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/
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
	export chnum=4
        export MIETABLE_DIR=${rttov_dir}/rtcoef_rttov12/mietable
elif [[ "$instrument" == "MHS" ]]; then
	export rtcoef_dir=${rttov_dir}/rtcoef_rttov11/rttov7pred54L
	export rtcoef=rtcoef_noaa_18_mhs.dat
elif [[ "$instrument" == "GIIRS" ]]; then
	export rtcoef_dir=${rttov_dir}/rtcoef_rttov11/rttov7pred101L
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
