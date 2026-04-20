#!/bin/sh
#BSUB -J CYCLE_DAKC
#BSUB -q serial
#BSUB -n 1
#BSUB -oo fkc.out
#BSUB -eo fkc.err
#scripts for cycling DA

#set -euo pipefail
export python_bin=/share/home/lililei1/kcfu/anaconda/envs/wrf/bin
export base_dir=/share/home/lililei1/kcfu/tc_mangkhut
export scratch_dir=/scratch/lililei1/kcfu/tc_mangkhut
export cycling_dir=${base_dir}/5cyclingDA
export initial_ensmem_dir=/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/10_00_00
export DART_DIR=/share/home/lililei1/kcfu/models/DART_main/models/wrf/work
export WRF_DIR=/share/home/lililei1/kcfu/models/real_WRF_WPS/V4.1/WRF-4.1
#==========================================================
#parameter settings
YEAR_MONTH="2018-09"
start_day=10
start_hour=00
start_min=00

end_day=10
end_hour=06
end_min=00

runtime_hour=6
runtime_minute=0

cycle_interval=6 #unit:hour
export max_dom=2

export filter_kind=1         #1=EAKF,2=QCF_RHF
export rttov_scatt=0         #0= simple cloud , 1 = rttov-scatt
export obs_err_std=0.25       #controls sigma_y
if [ "$filter_kind" -eq 1 ]; then
  echo "using EAKF"
  export post_anal_base_dir=$scratch_dir/5cyclingDA/postAnal_EAKF
elif [ "$filter_kind" -eq 2 ];then
  echo "using QCF_RHF"
  export post_anal_base_dir=$scratch_dir/5cyclingDA/postAnal_QCF_RHF
fi

#==========================================================
printf -v start_day_padded "%02d" $start_day
printf -v start_hour_padded "%02d" $start_hour
printf -v start_min_padded "%02d" $start_min

printf -v end_day_padded "%02d" $end_day
printf -v end_hour_padded "%02d" $end_hour
printf -v end_min_padded "%02d" $end_min

START_TIME_STR="${YEAR_MONTH}-${start_day_padded} ${start_hour_padded}:${start_min_padded}:00"
END_TIME_STR="${YEAR_MONTH}-${end_day_padded} ${end_hour_padded}:${end_min_padded}:00"

# interval seconds of cycle, unit:second
interval_seconds=$((6 * 60 * 60))

cycling_std_ts=$(date +%s -d "2018-09-10 00:00:00" 2>/dev/null)   #standard beginning time stamp
start_ts=$(date +%s -d "$START_TIME_STR" 2>/dev/null)
end_ts=$(date +%s -d "$END_TIME_STR" 2>/dev/null)
#===========================================================
# cycle
echo "cycling DA begins..."
export cycle_flag=1
cycle_count=0
for ((current_ts = start_ts; current_ts <= end_ts; current_ts += interval_seconds)); do
  export cycle_count=$((cycle_count + 1))
  #---------------------------------------------------------
    #cycle step 0 : calculate current time of cycle

  export current_day=$(date -d "@$current_ts" "+%d")
  export current_hour=$(date -d "@$current_ts" "+%H")
  export current_min=$(date -d "@$current_ts" "+%M")
  export current_time=${current_day}_${current_hour}_${current_min}

  export post_anal_dir=$post_anal_base_dir/d01_${current_time}
  mkdir -p ${post_anal_dir}
  echo "${post_anal_dir} is used to save analysis"
  export next_ts=$((current_ts + interval_seconds))
  export next_day=$(date -d "@$next_ts" "+%d")
  export next_hour=$(date -d "@$next_ts" "+%H")
  export next_min=$(date -d "@$next_ts" "+%M")
  
  export next_time=${next_day}_${next_hour}_${next_min}
  #mkdir -p ${post_anal_dir}

  if [ "$filter_kind" -eq 1 ]; then
  export cycle_wrfout_dir=${scratch_dir}/4assimilation/0mem_all_time/cyclingDA/${current_time}/EAKF
  mkdir -p ${cycle_wrfout_dir}
elif [ "$filter_kind" -eq 2 ];then
  export cycle_wrfout_dir=${scratch_dir}/4assimilation/0mem_all_time/cyclingDA/${current_time}/QCF_RHF
  mkdir -p ${cycle_wrfout_dir}
fi
  
  #-----------------------------------------------------------
  #cycle step 1 : create synthetic obs

  bash $base_dir/3create_obs/hx_rttov/run_rttov_TrueObs_driver_cyclingDA.sh > truthobs.out.${current_time} # "true" obs

  if (( current_ts == cycling_std_ts )); then
    #if this is the first cycle
    # export ens_wrfout_dir=/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/${start_day}_${start_hour}_${start_min}
    
    export profile_matlab_flag=1  #flag for wrf_rttov_d01_75km_ens.m to judge where the ensemble is
    echo "first cycle,use wrfout as prior"
    bash $base_dir/3create_obs/hx_rttov/run_rttov_ensBT_driver.sh > enshx.out.${current_time}
    
    
    # export ens_wrfout_dir=/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/${start_day}_${start_hour}_${start_min}
  else
    export profile_matlab_flag=2
    export ens_wrfout_dir=${cycle_wrfout_dir} #the two path should be same
    echo "not the first cycle, use cycling wrfout as prior"
    bash $base_dir/3create_obs/hx_rttov/run_rttov_ensBT_driver_cyclingDA.sh > enshx.out.${current_time}
    
  fi
  

  
  
  #-----------------------------------------------------------
    #cycle step 2 : do DA
    #step 2.1 convert obs
  if (( current_ts == cycling_std_ts )); then
    ${python_bin}/python $base_dir/4assimilation/1convert_obs/merge_FO_in_one_file.py
  else
    ${python_bin}/python $base_dir/4assimilation/1convert_obs/merge_FO_in_one_file_cycling.py
  fi


  if [ "$rttov_scatt" -eq 1 ];then
    ${python_bin}/python $base_dir/4assimilation/1convert_obs/obs2DART_rttov_scatt.py
  elif [ "$rttov_scatt" -eq 0 ];then
    ${python_bin}/python $base_dir/4assimilation/1convert_obs/obs2DART.py
  fi

  cd $base_dir/4assimilation/1convert_obs/run_dir
  export obs_seq_name=obs_seq.out_kctest1_d01_${current_time}_quantile
  sed -i "s#text_input_file.*#text_input_file = '${base_dir}/4assimilation/1convert_obs/obs_d01/kctest_676_obsinput_${current_time}_quantile.txt',#g" input.nml
  sed -i "s#FO_input_file.*#FO_input_file = '${base_dir}/4assimilation/1convert_obs/mergeFOd01_${current_time}/merged_FO_data.txt',#g" input.nml
  sed -i "s#obs_out_file.*#obs_out_file = ${obs_seq_name},#g" input.nml
  ./text_to_obs
  
  echo "convert obs done!"
    #step 2.2 DA
  cd ${base_dir}/4assimilation/2DART
  bsub < ${base_dir}/4assimilation/2DART/driver_DART_cyclingDA.sh #> ${cycling_dir}/DART.out.${current_time} 2>&1
  #if DART script is done, a DART_done_cycling file would be created
  elapsed_time=0
  max_wait_time=3600
  while [ ! -f "${base_dir}/4assimilation/2DART/run_dir/DART_done_cycling_${cycle_count}" ]; do
    if [ "$elapsed_time" -ge "$max_wait_time" ]; then
      echo "DART time out"
      exit 999
    fi
    sleep 30
    elapsed_time=$((elapsed_time + 30))
    
  done
  echo "========================================================"
  echo "                  DART assimilation done                "
  #--------------------------------------------------------------
  #cycle step 3 : run wrf to forcast
  #--------------------------------------------------------------
  # cycle step 3 : run wrf to forcast (Job Array + Smart Resume)
  #--------------------------------------------------------------
  if (( current_ts < end_ts )); then
    
    MAX_ATTEMPTS=3
    MIN_SIZE_BYTES=100000000
    declare -a success_mem

    # >>> 新增：阶段 0 (预检查 - 断点续跑逻辑) <<<
    echo "=========================================================="
    echo "Pre-checking completed members for cycle ${current_time}..."
    for ((imem=1;imem<=50;imem++)); do
      mem=$(printf "%03d" $imem)
      cycle_run_wrf_dir=${base_dir}/5cyclingDA/run_wrf/${current_time}/${mem}
      TARGET_FILE="${cycle_run_wrf_dir}/wrfout_d01_2018-09-${next_day}_${next_hour}:${next_min}:00"
      
      # 同时满足：1.当前时次的标志文件存在 2.输出文件存在 3.输出文件大于最小阈值
      if [ -f "${cycle_run_wrf_dir}/wrf_done_${current_time}" ] && [ -f "$TARGET_FILE" ]; then
         actual_size=$(stat -c%s "$TARGET_FILE")
         if [ "$actual_size" -ge "$MIN_SIZE_BYTES" ]; then
             echo "mem${mem} already successfully completed in a previous run. Skipping..."
             success_mem[$imem]=true
         else
             success_mem[$imem]=false
         fi
      else
         success_mem[$imem]=false
      fi
    done

    # 提前建立最终存放输出的目录路径
    if [ "$filter_kind" -eq 1 ]; then
      export cycle_wrfout_dir=${scratch_dir}/4assimilation/0mem_all_time/cyclingDA/${next_time}/EAKF
    elif [ "$filter_kind" -eq 2 ];then
      export cycle_wrfout_dir=${scratch_dir}/4assimilation/0mem_all_time/cyclingDA/${next_time}/QCF_RHF
    fi
    mkdir -p ${cycle_wrfout_dir}

    for attempt_num in $(seq 1 $MAX_ATTEMPTS); do
      pending_indices=""
      pending_count=0

      # >>> 阶段 1: 集中准备环境，并构建 LSF Array 索引序列 <<<
      for ((imem=1;imem<=50;imem++));do
        if [ "${success_mem[$imem]}" = false ]; then
          pending_count=$((pending_count + 1))
          mem=$(printf "%03d" $imem)
          cycle_run_wrf_dir=${base_dir}/5cyclingDA/run_wrf/${current_time}/${mem}
          
          if [ "$attempt_num" -eq 1 ]; then
            mkdir -p $cycle_run_wrf_dir
            cd ${cycle_run_wrf_dir}
            ln -sf ${WRF_DIR}/test/em_real/* .
            rm wrfinput_d0*
            #now we need to update wrfbdy_d01 after assimilation
            ln -sf ${base_dir}/1icbc/mem${mem}/BC/wrfbdy_d01_mem${mem} ./wrfbdy_d01_orig
            cp ${base_dir}/1icbc/mem${mem}/BC/wrfbdy_d01_mem${mem} ./wrfbdy_d01
            ln -sf ${base_dir}/0necessay_files/input.nml .
            ln -sf ${DART_DIR}/update_wrf_bc .
            #./update_wrf_bc > update_bc.out should link wrfinput first
            rm -f namelist.input
            cp ${base_dir}/0necessay_files/namelist.input.ens ./namelist.input
            
            sed -i "s#run_hours.*#run_hours = ${cycle_interval},#g" namelist.input
            sed -i "s#start_day.*#start_day = ${max_dom}*${current_day},#g" namelist.input
            sed -i "s#start_hour.*#start_hour = ${max_dom}*${current_hour},#g" namelist.input
            sed -i "s#start_minute.*#start_minute = ${max_dom}*${current_min},#g" namelist.input
            sed -i "s#max_dom.*#max_dom = ${max_dom},#g" namelist.input
            sed -i "s#input_from_file.*#input_from_file = .true.,.true.,.false.,#g" namelist.input
            cp ${base_dir}/0necessay_files/ad_omini_d01.py .
            cp ${base_dir}/0necessay_files/ad_omini_d02.py .
            for ((dom = 1; dom <= ${max_dom}; dom += 1)); do
              # make sure post_anal_dir files are ready
              min_size=100000000
              while true;do
                 target_file="${post_anal_dir}/analysis_d0${dom}.mem${mem}"
                 if [ -f "$target_file" ];then
                    actual_size=$(stat -c%s "$target_file")
                    if [ "$actual_size" -gt "$min_size" ];then
                      break
                    fi
                 fi
                 sleep 30
               
              done
              cp ${post_anal_dir}/analysis_d0${dom}.mem${mem} ./wrfinput_d0${dom}
              #ln -sf ${base_dir}/1icbc/mem${mem}/IC/wrfinput_201809${current_day}${current_hour}_mem${mem} ./wrfinput_d01_gfs
            done
            ./update_wrf_bc > update_bc.out
            ln -sf ${base_dir}/1icbc/mem${mem}/IC/wrfinput_201809${current_day}${current_hour}_mem${mem} ./wrfinput_d01_gfs
            ${python_bin}/python ./ad_omini_d01.py
            ${python_bin}/python ./ad_omini_d02.py
          fi

          # 【关键修改】清理旧时代的残留物，确保本次运行不受干扰
          cd ${cycle_run_wrf_dir}
          rm -f wrf_done* wrfout_d0*

          
          if [ -z "$pending_indices" ]; then
            pending_indices="$imem"
          else
            pending_indices="${pending_indices},${imem}"
          fi
        fi
      done

      # 如果没有待运行的成员（比如预检查发现 50 个全部跑完了，或者重试时都成功了），直接跳出
      if [ "$pending_count" -eq 0 ]; then
         echo "All members are ready. No WRF jobs need to be submitted for attempt $attempt_num."
         break
      fi

      # >>> 阶段 2: 一键提交 LSF 作业数组 <<<
      echo "=========================================================="
      echo "Attempt #$attempt_num: Submitting WRF Job Array for $pending_count members: [${pending_indices}]"
      cd ${base_dir}/5cyclingDA
      
      bsub -J "wrf_cyc_${current_time}_[${pending_indices}]" < ${base_dir}/0necessay_files/sub_wrf_array.sh
      
      # >>> 阶段 3: 集中监控所有运行中的成员 <<<
      echo "Waiting for array jobs to finish..."
      elapsed_time=0
      max_wait_time=28800 

      while true; do
        all_done=true
        for ((imem=1;imem<=50;imem++));do
          if [ "${success_mem[$imem]}" = false ]; then
            mem=$(printf "%03d" $imem)
            cycle_run_wrf_dir=${base_dir}/5cyclingDA/run_wrf/${current_time}/${mem}
            # 【关键修改】监控带有当前时间戳的标志文件
            if [ ! -f "${cycle_run_wrf_dir}/wrf_done_${current_time}" ]; then
              all_done=false
              break 
            fi
          fi
        done

        if [ "$all_done" = true ]; then
          echo "All jobs in current array generated wrf_done_${current_time}."
          break
        fi

        if [ "$elapsed_time" -ge "$max_wait_time" ]; then
          echo "WRF wait time out! Killing hanging jobs..."
          bkill -J "wrf_cyc" 2>/dev/null
          break 
        fi

        sleep 30
        elapsed_time=$((elapsed_time + 30))
      done

      # >>> 阶段 4: 检查文件大小并更新成功标志 <<<
      echo "Checking WRF outputs for attempt $attempt_num..."
      for ((imem=1;imem<=50;imem++));do
        if [ "${success_mem[$imem]}" = false ]; then
          mem=$(printf "%03d" $imem)
          cycle_run_wrf_dir=${base_dir}/5cyclingDA/run_wrf/${current_time}/${mem}
          TARGET_FILE="${cycle_run_wrf_dir}/wrfout_d01_2018-09-${next_day}_${next_hour}:${next_min}:00"

          if [ -f "$TARGET_FILE" ]; then
             actual_size=$(stat -c%s "$TARGET_FILE")
             if [ "$actual_size" -ge "$MIN_SIZE_BYTES" ]; then
                 echo "mem${mem}: SUCCESS"
                 success_mem[$imem]=true 
             else
                 echo "mem${mem}: FAILURE (Size $actual_size < $MIN_SIZE_BYTES)"
             fi
          else
             echo "mem${mem}: FAILURE (File not found)"
          fi
        fi
      done
      
    done 

    # >>> 阶段 5: 文件转移与清理 <<<
    echo "=========================================================="
    failed_members=""
    for ((imem=1;imem<=50;imem++));do
      mem=$(printf "%03d" $imem)
      if [ "${success_mem[$imem]}" = true ]; then
        cycle_run_wrf_dir=${base_dir}/5cyclingDA/run_wrf/${current_time}/${mem}
        cd ${cycle_run_wrf_dir}
        for ((dom = 1; dom <= ${max_dom}; dom += 1)); do
          # 注意：如果是预检查跳过的 member，它的 wrfout 可能上一次就已经被 mv 走了
          # 所以这里加一个 [ -f ] 的判断，避免重跑时报错 "mv: cannot stat"
          if [ -f "wrfout_d0${dom}_2018-09-${next_day}_${next_hour}:${next_min}:00" ]; then
             rm -f ${cycle_wrfout_dir}/firstguess_d0${dom}.mem${mem}
             mv wrfout_d0${dom}_2018-09-${next_day}_${next_hour}:${next_min}:00 ${cycle_wrfout_dir}/firstguess_d0${dom}.mem${mem}
             echo "wrfout${mem} has been moved to ${cycle_wrfout_dir}"
          fi
        done
        rm -f wrfout* wrfrst*
      else
        failed_members="$failed_members mem$mem"
      fi
    done
    for ((dom = 1; dom <= ${max_dom}; dom += 1)); do
    	ncea -O ${cycle_wrfout_dir}/firstguess_d0${dom}.mem* ${cycle_wrfout_dir}/firstguess_d0${dom}.ensmean
    done
    if [ ! -z "$failed_members" ]; then
      echo "FINAL FAILURE: The following members failed after $MAX_ATTEMPTS attempts: $failed_members"
      exit 1
    fi

    echo "All 50 WRF ensemble members completed successfully for cycle ${current_time}."
    echo "--------------------"
  fi
#--------------------------------------ensemble forecast done--------------------------------------------------------------- 
done
echo "===================="
echo "cycle done"

