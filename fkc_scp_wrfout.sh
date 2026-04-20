#!/bin/sh
export WORK_DIR=/share/home/lililei1/kcfu/tc_mangkhut
export ENS_WRF_DIR=${WORK_DIR}/2ens_free_fcst
export DATA_STORAGE_DIR=/ldata3/llleistu/kcfu/tc_mangkhut
maxdom=1
memlist=(24 25 39 31 38 73 49 41 62)
#memlist=(51 70 54 48 47 69 52 20 71)
for imem in ${memlist[*]};do

	member="mem"`printf %03i ${imem}`
	echo "scp ing ${member}"
	mem_dir=${ENS_WRF_DIR}/${member}
	idom=1
	while [[ $idom -le $maxdom ]];do
		dom="d"`printf %02i ${idom}`
		sshpass -p 'lilic503' ssh llleistu@114.212.48.200 "mkdir -p ${DATA_STORAGE_DIR}/ensmem/${member}/${dom}"
		sshpass -p 'lilic503' scp ${mem_dir}/wrfout_${dom}* llleistu@114.212.48.200:${DATA_STORAGE_DIR}/ensmem/${member}/${dom}/
		idom=`expr ${idom} \+ 1`
	done
	
done
