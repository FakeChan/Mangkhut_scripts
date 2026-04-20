#! /bin/bash
#20201017 在每个文件夹里生成BT obs
#time='04_01_50'
#chnum=1650
######################################## 1-70channel  24*70
#domain=d02
#rundir=/share/home/lililei1/lfzhou/hyperspectral_da/step1_obs_ensBT/step2_les_obs/4obsdir_$domain
#bt_channel_dir=$rundir/BT_$time
time=${obs_day}_${obs_hour}_${obs_min}
bt_channel_dir=${obs_mem_dir}/${instrument}/BT_${time}
mkdir -p $bt_channel_dir
# mkdir -p $bt_channel_dir/pics_obs_withpert
# mkdir -p $bt_channel_dir/pics_obsnr
		cd ${obs_dir}/merge
		rm run_hebing.sh
		cat >> run_hebing.sh << EOF
#!/bin/sh
#=================================================================
#BSUB -J hebing_obs
#BSUB -n 1
#BSUB -o hebing_obs$i.log
#BSUB -e hebing_obs$i.err
#=================================================================
matlab -nodesktop -nosplash -nodisplay < hebing_diffchan.m
EOF
chmod +x run_hebing.sh
	./run_hebing.sh
#bsub < run_hebing.sh            #./run(n)#############################
#sleep 600

