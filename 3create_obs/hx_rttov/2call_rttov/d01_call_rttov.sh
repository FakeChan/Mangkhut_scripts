#!/bin/sh
# rttov & calc Pmax
time=${obs_day}_${obs_hour}_${obs_min}
#call_rttov_dir=/share/home/lililei1/lfzhou/hyperspectral_da/step1_obs_ensBT/step2_les_obs/2call_rttov_$domain
#prof_dir=/share/home/lililei1/lfzhou/hyperspectral_da/step1_obs_ensBT/step2_les_obs/3profdata_$domain
#obs_dir=/share/home/lililei1/lfzhou/hyperspectral_da/step1_obs_ensBT/step2_les_obs/4obsdir_$domain
#pmax_dir=/share/home/lililei1/lfzhou/hyperspectral_da/step1_obs_ensBT/step4_Pmax/calcPmax_fwd_$domain
export call_rttov_dir=${work_dir}/2call_rttov
if [ $rttov_scatt -eq 1 ];then
    running_script_file=run_les_676point_fwd_rttov_scatt.sh
    sed -i "s/MIETABLE_FILENAME=.*$/MIETABLE_FILENAME\=\"$MIETABLE\"/g"           $call_rttov_dir/${running_script_file}
else
    running_script_file=run_les_676point_fwd.sh
fi

cd    $call_rttov_dir
mkdir -p ${call_rttov_dir}/call_rttov_test ${prof_mem_dir}/${instrument}/ ${obs_mem_dir}/${instrument}
#sed -i "s/TEST\=.*$/TEST\=\.\/2call_rttov_$domain/g" $call_rttov_dir/run_les_676point_fwd.sh
sed -i "s/NCHAN=.*$/NCHAN\=$chnum/g"           $call_rttov_dir/${running_script_file}
sed -i "s/NPROF=.*$/NPROF\=$npoint/g"          $call_rttov_dir/${running_script_file}
sed -i "s/NLEVELS=.*$/NLEVELS\=$nlevels/g"          $call_rttov_dir/${running_script_file}
sed -i "s/\=\"prof.*$/\=\"prof${time}.dat\"/g" $call_rttov_dir/${running_script_file}
sed -i "s/COEF_FILENAME=.*$/COEF_FILENAME\=\"${rtcoef}\"/g"          $call_rttov_dir/${running_script_file}
# sed -i "s/DATART=.*$/DATART\=$rtcoef_dir/g"          $call_rttov_dir/run_les_676point_fwd.sh
echo $time

ln -sf ${prof_mem_dir}/prof${obs_day}_${obs_hour}:${obs_min}.dat ${call_rttov_dir}/call_rttov_test/prof${time}.dat
$call_rttov_dir/${running_script_file} ARCH=gfortran

if [ $rttov_scatt -eq 0 ];then
    mv $call_rttov_dir/call_rttov_test/prof${time}.dat.gfortran ${prof_mem_dir}/${instrument}/ens${member}_${domain}_output${time}.dat
    #sed -n '49,99999999p' ${prof_mem_dir}/${instrument}/ens${member}_${domain}_output${time}.dat > ${obs_mem_dir}/${instrument}/ens${member}_${domain}_output_${time}.txt
    export BT_output_dir=${prof_mem_dir}/${instrument}
    export BT_input=ens${member}_${domain}_output${time}.dat
    export BT_output=ens${member}_${domain}_output_${time}.txt
    export BT_write_dir=${obs_mem_dir}
    ${python_bin}/python ${work_dir}/2call_rttov/extractBT_rttov_scatt.py
else
    export BT_output_dir=${prof_mem_dir}/${instrument}
    export BT_input=ens${member}_${domain}_output${time}.dat
    export BT_output=ens${member}_${domain}_output_${time}.txt
    export BT_write_dir=${obs_mem_dir}
    mv $call_rttov_dir/call_rttov_test/output_example_rttovscatt_fwd.dat ${BT_output_dir}/${BT_input}
    ${python_bin}/python ${work_dir}/2call_rttov/extractBT_rttov_scatt.py
fi
exit

#calc Pmax
# cd    $pmax_dir
# sed -i "s/TEST\=.*$/TEST\=\.\/calcPmax_fwd_$domain/g"    run_prof_checkhx_calcPmax_fwd.sh
# sed -i "s/NCHAN\=.*$/NCHAN\=$chnum/g"           run_prof_checkhx_calcPmax_fwd.sh
# sed -i "s/NPROF\=$npoint/NPROF\=$npoint/g"     run_prof_checkhx_calcPmax_fwd.sh
# sed -i "s/\=\"prof.*$/\=\"prof${time}.dat\"/g" run_prof_checkhx_calcPmax_fwd.sh
# echo $time

# ln -sf $prof_dir/prof${time}.dat . 
# sed -n '49,99999999999p' prof${time}.dat.gfortran >> prof${time}datgfortran.txt
# sed -i "s/chnum\=.*$/chnum\=${chnum}/g" diff_Pmax_676point.m
# sed -i "s/npoint\=.*$/npoint\=$npoint/g" diff_Pmax_676point.m
# sed -i "s/time\=.*$/time\=\'$time\'/g" diff_Pmax_676point.m
# ./bsub_Pmax.sh
# echo "Pmax calc done"
