#!/bin/sh
#=================================================================
#BSUB -J runmatlab_n_6min
#BSUB -n 1
#BSUB -q normal 
#BSUB -o runmatlab_n.log
#BSUB -e runmatlab_n.err
#BSUB -R "span[ptile=20]"
#=================================================================
if [ $rttov_scatt -eq 1 ];then
    running_script_file=wrf_rttov_d01_75km_ens_rttov_scatt.m
else
    running_script_file=wrf_rttov_d01_75km_ens.m
fi
matlab -nodesktop -nosplash -nodisplay <  ${running_script_file}
