#!/bin/sh

#BSUB -J ensfcst_gp2
#BSUB -q largemem
#BSUB -n 144
#BSUB -oo freefcst.out2
#BSUB -eo freefcst.err2


export WRF_DIR=/share/home/lililei1/kcfu/models/real_WRF_WPS/V4.1/WRF-4.1/test/em_real

export WORK_DIR=/share/home/lililei1/kcfu/tc_mangkhut   #where this script located
export NECESSAY_FILE_DIR=${WORK_DIR}/0necessay_files
export ICBC_DIR=${WORK_DIR}/1icbc
export ENS_WRF_DIR=${WORK_DIR}/2ens_free_fcst #where ensemble members are stored and forcasted
export SCRATCH_DIR=/scratch/lililei1/kcfu/tc_mangkhut/2ens_free_fcst
fcst_range=24

STARTyy=2018
STARTmm=09

STARTdd_d01=09
STARTdd_d02=09

STARThh_d01=00
STARThh_d02=06
STARTmin=00


ENDyy=2018
ENDmm=09

ENDdd_d01=10
ENDdd_d02=10

ENDhh_d01=00
ENDhh_d02=00
ENDmin=00

max_dom=2

STARTtime=${STARTyy}${STARTmm}${STARTdd_d01}${STARThh_d01}${STARTmin}
start_time_wrf=`${NECESSAY_FILE_DIR}/da_advance_time.exe ${STARTtime} 0 -w`
start_time=`${NECESSAY_FILE_DIR}/da_advance_time.exe ${STARTtime} 0`

end_time_wrf=`${NECESSAY_FILE_DIR}/da_advance_time.exe ${STARTtime} ${fcst_range} -w`
# echo $start_time

#memlist=(51 70 54 48 47 69 52 20 71)
#memlist=(24 25 39 31 38 73 49 41 62)
# memlist=(39 62)
# rm ${ENS_WRF_DIR}/wrf/*
# ln -sf ${WRF_DIR}/* ${ENS_WRF_DIR}/wrf/

# for imem in ${memlist[*]};do
for (( imem=17; imem<=20; imem++ ));do
	# imem=$LSB_JOBINDEX
	member="mem"`printf %03i ${imem}`
	mkdir -p ${ENS_WRF_DIR}/${member}
	mkdir -p ${SCRATCH_DIR}/${member}
	if [ -e "${ENS_WRF_DIR}/${member}/wrfout_d01_${end_time_wrf}" ]; then
		# echo "${member} done. Continue"
		continue
	else
		echo "${member} running"
		ln -sf ${WRF_DIR}/* ${ENS_WRF_DIR}/${member}/
		#cp ${ENS_WRF_DIR}/wrf/* ${ENS_WRF_DIR}/${member}/
		cd ${ENS_WRF_DIR}/${member}
		cp ${NECESSAY_FILE_DIR}/namelist.input.ens ./namelist.input

		#modify code and namelist
		# sed -i "s/filepath=.*/filepath = \".\/\"/g" 1assign_d01.ncl
		# sed -i "s/filepath=.*/filepath = \".\/\"/g" 2assign_d02.ncl

		sed -i "s/run_hours.*/run_hours =                             ${fcst_range},/g" namelist.input
		sed -i "s/start_day.*/start_day =                             ${STARTdd_d01},${STARTdd_d02},/g" namelist.input
		sed -i "s/end_day.*/end_day =                             ${ENDdd_d01},${ENDdd_d02},/g" namelist.input
		sed -i "s/start_hour.*/start_hour =                             ${STARThh_d01},${STARThh_d02}/g" namelist.input
		sed -i "s/end_hour.*/end_hour =                             ${ENDhh_d01},${ENDhh_d02}/g" namelist.input
		sed -i "s/max_dom.*/max_dom                     = ${max_dom}/g" namelist.input
		#link wrfinput and wrfbdy
		ln -sf ${ICBC_DIR}/${member}/IC/wrfinput_${start_time}_${member} ${ENS_WRF_DIR}/${member}/wrfinput_d01
		#ln -sf ${NECESSAY_FILE_DIR}/wrfout_d02_${start_time_wrf} ${ENS_WRF_DIR}/${member}/wrfinput_d02
		ln -sf ${ICBC_DIR}/${member}/BC/wrfbdy_d01_${member} ${ENS_WRF_DIR}/${member}/wrfbdy_d01
                #ln -sf ${NECESSAY_FILE_DIR}/wrfbdy_d01 ${ENS_WRF_DIR}/${member}/wrfbdy_d01

		# ncl 1assign_d01.ncl
		# ncl 2assign_d02.ncl
		export I_MPI_DEVICE=rdssm
		ulimit -s unlimited
		ulimit -d unlimited
		export WRFIO_NCD_LARGE_FILE_SUPPORT=1
		mpiexec.hydra ./wrf.exe
		#mpirun -np 80 ./wrf.exe
		mv rsl.error.0000 rsl.err
		mv rsl.out.0000 rsl.out

		rm rsl.error*
		rm rsl.out.*

		#move wrfout to scratch
		mv wrfout_d0* ${SCRATCH_DIR}/${member}
		mv wrfrst_d0* ${SCRATCH_DIR}/${member}
	fi
done
