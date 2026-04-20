#!/bin/sh


datadir_in=/share/home/lililei1/kcfu/tc_mangkhut/2ens_free_fcst
#setenv datadir_out    /ldata4/llleistu/jjdeng/hz2023/exp_hybrid_v2
datadir_out=/ldata3/llleistu/kcfu/tc_mangkhut/ensmem

# setenv TOOL_DIR       /share/home/lililei/models/WRFDA/var/da # Location of da_advance_time.exe 

# setenv DATE $start_init

for (( imem=1; imem<=50; imem++ ));do
  member="mem"`printf %03i ${imem}`
  # # copy fc
  # cd $datadir_in/$DATE/fc/enkf_ens_mean
  # #/share/apps.20191030bak/sshpass-1.06/bin/sshpass -p 'lilic503' ssh llleistu@114.212.48.200 "mkdir -p $datadir_out/$DATE/fc/enkf_ens_mean/"
  # #/share/apps.20191030bak/sshpass-1.06/bin/sshpass -p 'lilic503' scp namelist.input ./wrfout_d01* llleistu@114.212.48.200:$datadir_out/$DATE/fc/enkf_ens_mean/
  # /share/apps.20191030bak/sshpass-1.06/bin/sshpass -p 'lilic503' ssh lfzhou@114.212.48.217 "mkdir -p $datadir_out/$DATE/fc/enkf_ens_mean/"
  # /share/apps.20191030bak/sshpass-1.06/bin/sshpass -p 'lilic503' scp namelist.input rsl.out_SAVE ./wrfout_d01* ./wrfout_d02* lfzhou@114.212.48.217:$datadir_out/$DATE/fc/enkf_ens_mean/

  # cd $datadir_in/$DATE/fc/hybrid_ens_mean
  # /share/apps.20191030bak/sshpass-1.06/bin/sshpass -p 'lilic503' ssh lfzhou@114.212.48.217 "mkdir -p $datadir_out/$DATE/fc/hybrid_ens_mean/"
  # /share/apps.20191030bak/sshpass-1.06/bin/sshpass -p 'lilic503' scp namelist.input rsl.out_SAVE ./wrfout_d01* ./wrfout_d02* lfzhou@114.212.48.217:$datadir_out/$DATE/fc/hybrid_ens_mean/

  # cd $datadir_in/$DATE/fc/ens_1
  # /share/apps.20191030bak/sshpass-1.06/bin/sshpass -p 'lilic503' ssh lfzhou@114.212.48.217 "mkdir -p $datadir_out/$DATE/fc/ens_1/"
  # /share/apps.20191030bak/sshpass-1.06/bin/sshpass -p 'lilic503' scp namelist.input rsl.out_SAVE ./wrfout_d01* ./wrfout_d02* lfzhou@114.212.48.217:$datadir_out/$DATE/fc/ens_1/

  # cd $datadir_in/$DATE/fc/ens_2
  # /share/apps.20191030bak/sshpass-1.06/bin/sshpass -p 'lilic503' ssh lfzhou@114.212.48.217 "mkdir -p $datadir_out/$DATE/fc/ens_2/"
  # /share/apps.20191030bak/sshpass-1.06/bin/sshpass -p 'lilic503' scp namelist.input rsl.out_SAVE ./wrfout_d01* ./wrfout_d02* lfzhou@114.212.48.217:$datadir_out/$DATE/fc/ens_2/
  if [ -e "${datadir_in}/${member}/wrfout_d01_2018-09-13_00:00:00" ]; then
    echo "${member} scp ing"
    cd $datadir_in/${member}
    /share/apps.20191030bak/sshpass-1.06/bin/sshpass -p 'lilic503' ssh llleistu@114.212.48.200 "mkdir -p $datadir_out/${member}/d01"
    /share/apps.20191030bak/sshpass-1.06/bin/sshpass -p 'lilic503' scp namelist.input rsl.out* ./wrfout_d01* llleistu@114.212.48.200:$datadir_out/${member}/d01
    
  else
    echo "${member} not done"
    continue

  # setenv DATE `$TOOL_DIR/da_advance_time.exe $DATE $CYCLE_PERIOD`   #advance date
  fi
done
