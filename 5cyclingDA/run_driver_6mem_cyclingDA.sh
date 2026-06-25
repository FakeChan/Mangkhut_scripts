#!/bin/sh
#this is a very simple , 6 mem ensemble test for cycling DA. 
#It's based on you have already run a complete cycle using driver.
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
memlist=(6 15 29 37 43 44)
ensmem_dir=/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/cyclingDA/10_00_00
run_wrf_dir=/share/home/lililei1/kcfu/tc_mangkhut/5cyclingDA/run_wrf/10_00_00
YEAR_MONTH="2018-09"
start_day=10
start_hour=00
start_min=00

end_day=10
end_hour=06
end_min=00

runtime_hour=6
runtime_minute=0

current_day=10
current_hour=00
current_min=00
export current_time=10_00_00

next_day=10
next_hour=06
next_min=00
next_time=10_06_00
cycle_interval=6 #unit:hour
export max_dom=2

export filter_kind=2         #1=EAKF,2=QCF_RHF
export rttov_scatt=0         #0= simple cloud , 1 = rttov-scatt
export obs_err_std=0.25       #controls sigma_y

export update_ocean=0  #0 = no update ocean
export run_ocean=0     #1= run wrf with ocean

run_wrf_flag=1
run_da_flag=1

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

if [ "$run_da_flag" -eq 1 ];then
  #---------------------------------------
  #             first do DA              -
  #---------------------------------------
  cd ${base_dir}/4assimilation/2DART/run_dir  #enter DART running dir

  #----------------------------------------
  #step 1: if update_ocean == 1, first inflate ocean and update
  if [ "$update_ocean" -eq 1 ];then
    cp ./input.nml.inflateOcean ./input.nml.step1
    if [ "$filter_kind" -eq 1 ]; then
        echo "using EAKF"
      elif [ "$filter_kind" -eq 2 ];then
        sed -i "s/qceff_table_filename[[:space:]]*=.*/qceff_table_filename  =   'qceff_table_fkc.csv',/g" input.nml.step1
        echo "using QCF_RHF"
    fi
  
    cp input.nml.step1 input.nml
    #-------------make sure obs_seq.out is using LACC----------------
    rm obs_seq.out
    ln -sf /share/home/lililei1/kcfu/tc_mangkhut/4assimilation/1convert_obs/run_dir/obs_seq.out_kctest1_d01_10_00_00_LACC_ch4 ./obs_seq.out
    #----------------------------------------------------------------
    rm -f fkc_dart
    bsub < ./sub_dart.sh
    # mpirun ./filters
    elapsed_time=0
    max_wait_time=3600
    MIN_SIZE_BYTES=100000000
    # wait until output done
    while true;do
      target_file="${base_dir}/4assimilation/2DART/run_dir/output_d01.mem006"
      if [ -f "${base_dir}/4assimilation/2DART/run_dir/fkc_dart" ] && [ -f "$target_file" ] ;then
        actual_size=$(stat -c%s "$target_file")
        if [ "$actual_size" -ge "$MIN_SIZE_BYTES" ]; then
          echo "DART successfully done"
          break
        fi
      fi
      sleep 30
      elapsed_time=$((elapsed_time + 30))

      if [ "$elapsed_time" -ge "$max_wait_time" ]; then
          echo "DART time out"
          exit 999
      fi
    done

    mkdir -p ${base_dir}/4assimilation/2DART/run_dir/inflatedOcean
    mv ${base_dir}/4assimilation/2DART/run_dir/output_d0* ${base_dir}/4assimilation/2DART/run_dir/inflatedOcean
  fi
  #----------------------------------------
  #step2 update air components
  if [ "$update_ocean" -eq 1 ];then
    cp ./input.nml.all ./input.nml.step2
    if [ "$filter_kind" -eq 1 ]; then
        echo "using EAKF"
      elif [ "$filter_kind" -eq 2 ];then
        sed -i "s/qceff_table_filename[[:space:]]*=.*/qceff_table_filename  =   'qceff_table_fkc.csv',/g" input.nml.step2
        echo "using QCF_RHF"
    fi
  elif [ "$update_ocean" -eq 0 ];then
    cp ./input.nml.air ./input.nml.step2
    if [ "$filter_kind" -eq 1 ]; then
        echo "using EAKF"
      elif [ "$filter_kind" -eq 2 ];then
        sed -i "s/qceff_table_filename[[:space:]]*=.*/qceff_table_filename  =   'qceff_table_fkc.csv',/g" input.nml.step2
        echo "using QCF_RHF"
    fi
  fi
  rm -f fkc_dart
  cp ./input.nml.step2 ./input.nml

  rm obs_seq.out
  ln -sf /share/home/lililei1/kcfu/tc_mangkhut/4assimilation/1convert_obs/run_dir/obs_seq.out_kctest1_d01_10_00_00_quantile_ch4_clrsky ./obs_seq.out
  bsub < ./sub_dart.sh
  # mpirun ./filters
  elapsed_time=0
  max_wait_time=3600
  MIN_SIZE_BYTES=100000000
  # wait until output done
  while true;do
    target_file="${base_dir}/4assimilation/2DART/run_dir/output_d01.mem006"
    if [ -f "${base_dir}/4assimilation/2DART/run_dir/fkc_dart" ] && [ -f "$target_file" ] ;then
      actual_size=$(stat -c%s "$target_file")
      if [ "$actual_size" -ge "$MIN_SIZE_BYTES" ]; then
        echo "DART successfully done"
        break
      fi
    fi
    sleep 30
    elapsed_time=$((elapsed_time + 30))

    if [ "$elapsed_time" -ge "$max_wait_time" ]; then
        echo "DART time out"
        exit 999
    fi
  done

  #-------------------------------------------------
  #step3: if update_ocean ==1 , replace ouput with inflated ocean
  #!/bin/bash
  if [ "$update_ocean" -eq 1 ];then
    OUT_DIR="${base_dir}/4assimilation/2DART/run_dir"    # 存放 output_d01.mem* 的文件夹
    INF_DIR="${base_dir}/4assimilation/2DART/run_dir/inflatedOcean"   # 存放 inflate_d01.mem* 的文件夹

    echo "开始执行变量移植，将用 $INF_DIR 下的 OM_TMP 覆盖 $OUT_DIR 下的对应文件..."

    # 循环 1 到 50
    for i in {1..50}; do
        mem_id=$(printf "%03d" $i)
        
        out_file_d01="${OUT_DIR}/output_d01.mem${mem_id}"
        inf_file_d01="${INF_DIR}/output_d01.mem${mem_id}"

        out_file_d02="${OUT_DIR}/output_d02.mem${mem_id}"
        inf_file_d02="${INF_DIR}/output_d02.mem${mem_id}"
        
        # 检查两个文件是否同时存在
        if [[ -f "$out_file_d01" && -f "$inf_file_d01" ]]; then
            echo "Processing member ${mem_id}..."
            
            # 核心指令：
            # -A : 追加模式 (存在同名则覆盖)
            ncks -A -v OM_TMP,OM_S,OM_U,OM_V "$inf_file_d01" "$out_file_d01"
            ncks -A -v OM_TMP,OM_S,OM_U,OM_V "$inf_file_d02" "$out_file_d02"
            
        else
            echo "警告: 找不到成员 ${mem_id} 的对应文件，已跳过。"
        fi
    done

  fi
  #-------------------------------------------------
  for imem in "${memlist[@]}";do
    member="mem"`printf %03i ${imem}`
    idx=`printf %03i ${imem}`
    for ((dom = 1; dom <= ${max_dom}; dom += 1)); do
      cp ${ensmem_dir}/firstguess_d0${dom}.${member} ./firstguess_d0${dom}.${member}
      cp ./output_d0${dom}.${member} ./analysis_d0${dom}.${member}
      export base_file=firstguess_d0${dom}.${member}  #base_file is the file to be updated (fg)
      export updated_file=analysis_d0${dom}.${member}   #updated file is the file already updated after DART
      if [ "$update_ocean" -eq 1 ];then
        ./ncks.sh
      elif [ "$update_ocean" -eq 0 ];then
        ./ncks_air.sh
      fi
      # mv firstguess_d0${dom}.${member} ${run_wrf_dir}/$idx/wrfinput_d0${dom}
    done
  done
  rm analysis_d0*
  rm output_d0*
fi
#---------------------------------------
#             now run wrf              -
#---------------------------------------

if [ "$run_wrf_flag" -eq 1 ];then
  MAX_ATTEMPTS=3
  MIN_SIZE_BYTES=100000000
  declare -a success_mem

  # >>> 新增：阶段 0 (预检查 - 断点续跑逻辑) <<<
  echo "=========================================================="
  echo "Pre-checking completed members for cycle ${current_time}..."
  for imem in "${memlist[@]}"; do
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
    export cycle_wrfout_dir=${scratch_dir}/cycle_test/6mem_oceanAssim${update_ocean}Run${run_ocean}/EAKF
    
  elif [ "$filter_kind" -eq 2 ];then
    export cycle_wrfout_dir=${scratch_dir}/cycle_test/6mem_oceanAssim${update_ocean}Run${run_ocean}/QCF_RHF
    
  fi
  mkdir -p ${cycle_wrfout_dir}

  for attempt_num in $(seq 1 $MAX_ATTEMPTS); do
    pending_indices=""
    pending_count=0

    # >>> 阶段 1: 集中准备环境，并构建 LSF Array 索引序列 <<<
    for imem in "${memlist[@]}";do
      if [ "${success_mem[$imem]}" = false ]; then
        pending_count=$((pending_count + 1))
        mem=$(printf "%03d" $imem)
        cycle_run_wrf_dir=${base_dir}/5cyclingDA/run_wrf/${current_time}/${mem}
        
        if [ "$attempt_num" -eq 1 ]; then
          mkdir -p $cycle_run_wrf_dir
          cd ${cycle_run_wrf_dir}
          ln -sf ${WRF_DIR}/test/em_real/* .
          rm wrfinput_d0*
          for ((dom = 1; dom <= ${max_dom}; dom += 1)); do
            mv ${base_dir}/4assimilation/2DART/run_dir/firstguess_d0${dom}.mem${mem} ./wrfinput_d0${dom}
          done  
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
          sed -i "s#history_interval.*#history_interval                    = 30,  30,   30,#g" namelist.input
          sed -i "s#max_dom.*#max_dom = ${max_dom},#g" namelist.input
          sed -i "s#input_from_file.*#input_from_file = .true.,.true.,.false.,#g" namelist.input
          if [ "$run_ocean" -eq 1 ];then 
            sed -i "s#sf_ocean_physics.*#sf_ocean_physics         = 2,#g" namelist.input
          elif [ "$run_ocean" -eq 0 ];then
            sed -i "s#sf_ocean_physics.*#sf_ocean_physics         = 0,#g" namelist.input
          fi
          cp ${base_dir}/0necessay_files/ad_omini_d01.py .
          cp ${base_dir}/0necessay_files/ad_omini_d02.py .
          
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
      for imem in "${memlist[@]}";do
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
    for imem in "${memlist[@]}";do
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
  for imem in "${memlist[@]}";do
    mem=$(printf "%03d" $imem)
    if [ "${success_mem[$imem]}" = true ]; then
      cycle_run_wrf_dir=${base_dir}/5cyclingDA/run_wrf/${current_time}/${mem}
      mkdir -p $cycle_wrfout_dir/${mem}
      cd ${cycle_run_wrf_dir}
      mv wrfout_d0* $cycle_wrfout_dir/${mem}
      rm -f wrfout* wrfrst*
    else
      failed_members="$failed_members mem$mem"
    fi
  done

  if [ ! -z "$failed_members" ]; then
    echo "FINAL FAILURE: The following members failed after $MAX_ATTEMPTS attempts: $failed_members"
    exit 1
  fi

  echo "All WRF ensemble members completed successfully for cycle ${current_time}."
  echo "--------------------"
fi
  

