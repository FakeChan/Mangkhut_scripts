#!/bin/sh
#BSUB -q mpi
#BSUB -n 48
#BSUB -oo icbc.out
#BSUB -eo icbc.error
#BSUB -J icbc
export python_bin=/share/home/lililei1/kcfu/anaconda/envs/wrf/bin
export WORK_DIR=/share/home/lililei1/kcfu/tc_mangkhut   #where this script located

export NECESSAY_FILE_DIR=${WORK_DIR}/0necessay_files
export ICBC_DIR=${WORK_DIR}/1icbc

export DART_DIR=/share/home/lililei1/kcfu/models/DART_main/models/wrf/work
export TOP_WRFDA=/share/home/lililei1/kcfu/models/real_WRF_WPS/V4.1/DA/WRFDAV4
export WRFDA_DIR=${TOP_WRFDA}/var/build   #where da_wrfvar.exe located
export WRFDA_CV3_DIR=${TOP_WRFDA}/var/run #where the cv3 options located, i.e. be.dat.cv3, gribmap.txt, LANDUSE.TBL
export WRF_DIR=/share/home/lililei1/kcfu/models/real_WRF_WPS/V4.1/WRF-4.1/test/em_real



nmem=80 	#the number of all ensemble members
lbc_freq=6  #frequency to update wrfbdy
fcst_range=72

STARTyy=2018
STARTmm=09
STARTdd=09
STARThh=00
STARTmin=00
STARTtime=${STARTyy}${STARTmm}${STARTdd}${STARThh}${STARTmin}

next_hour=06

ENDyy=2018
ENDmm=09
ENDdd=12
ENDhh=00
ENDmin=00
max_dom=1

Nt=`expr $fcst_range \/ $lbc_freq \+ 1` #the number of LBC in forcast range.
ln -sf ${WRFDA_DIR}/da_advance_time.exe ${NECESSAY_FILE_DIR}
start_time_wrf=`${NECESSAY_FILE_DIR}/da_advance_time.exe ${STARTtime} 0 -w`
start_time_plus1_wrf=`${NECESSAY_FILE_DIR}/da_advance_time.exe ${STARTtime} ${lbc_freq} -w`

# ########################################################
# # 0. generate IC BC for NR
# ########################################################

# cd ${NECESSAY_FILE_DIR}
# sed -i "s/run_hours.*/run_hours =                             ${fcst_range},/g" namelist.wrf
# sed -i "s/start_day.*/start_day =                             ${max_dom}*${STARTdd},/g" namelist.wrf
# sed -i "s/end_day.*/end_day =                             ${max_dom}*${ENDdd},/g" namelist.wrf
# sed -i "s/start_hour.*/start_hour =                             ${max_dom}*${STARThh},/g" namelist.wrf
# sed -i "s/end_hour.*/end_hour =                             ${max_dom}*${ENDhh},/g" namelist.wrf
# cd ${WRF_DIR}
# ln -sf ${NECESSAY_FILE_DIR}/namelist.wrf ./namelist.input

# mpirun ./real.exe
# mv rsl.out.0000 rsl.out  #clean-up
# mv rsl.error.0000 rsl.error
# rm -f rsl.out.*
# rm -f rsl.error.*

# cp ./wrfbdy_d01 ${NECESSAY_FILE_DIR}
# rm ./wrfbdy_d01

# ########################################################
# # 1. generate IC BC for ensmean at each time
# ########################################################
# it=1
# while [[ ${it} -le ${Nt} ]];do
# 	dt=`expr \( ${it} \- 1 \) \* ${lbc_freq}`
# 	dt_plus_one=`expr \( ${it} \) \* ${lbc_freq}`
# 	this_time=`${NECESSAY_FILE_DIR}/da_advance_time.exe ${STARTtime} ${dt} `
# 	next_time=`${NECESSAY_FILE_DIR}/da_advance_time.exe ${STARTtime} ${dt_plus_one} `
# 	this_day=${this_time:6:2}
# 	next_day=${next_time:6:2}
# 	this_hour=${this_time:8:2}
# 	next_hour=${next_time:8:2}
# 	# echo ${this_day}
# 	# echo ${this_hour}
# # omit other options, only change day and hour
# 	cd ${NECESSAY_FILE_DIR}

# 	sed -i "s/start_day.*/start_day =                             ${max_dom}*${this_day},/g" namelist.input.ens
# 	sed -i "s/end_day.*/end_day =                             ${max_dom}*${next_day},/g" namelist.input.ens
# 	sed -i "s/start_hour.*/start_hour =                             ${max_dom}*${this_hour},/g" namelist.input.ens
# 	sed -i "s/end_hour.*/end_hour =                             ${max_dom}*${next_hour},/g" namelist.input.ens

# # 	sed -i "s/start_hour.*/start_hour =                             ${STARThh},/g" namelist.WRFDA.VAR
# # 	sed -i "s/end_hour.*/end_hour =                             ${ENDhh},/g" namelist.WRFDA.VAR
# # 	sed -i "s/analysis_date.*/analysis_date ="${start_time_wrf}.0000",/g" namelist.WRFDA.VAR

# 	cd ${WRF_DIR}
# 	rm -f namelist.input
# 	ln -sf ${NECESSAY_FILE_DIR}/namelist.input.ens ./namelist.input
# 	rm -f wrfinput_d01
# 	rm -f wrfbdy_d01
# 	mpirun ./real.exe
# 	mv rsl.out.0000 rsl.out  #clean-up
# 	mv rsl.error.0000 rsl.error
# 	rm -f rsl.out.*
# 	rm -f rsl.error.*

#  	cp ./wrfinput_d01 ${NECESSAY_FILE_DIR}/wrfinput_d01_${this_time}
# 	it=`expr ${it} \+ 1`
# done



########################################################
# 2. generate IC BCs
########################################################
imem=1
# while [[ ${imem} -le ${nmem} ]];do
while [[ ${imem} -le 50 ]];do
	it=1
	################################################
	# 2.1 first, ICs
	################################################
	while [[ ${it} -le ${Nt} ]];do
		dt=`expr \( ${it} \- 1 \) \* ${lbc_freq}`
		dt_plus_one=`expr \( ${it} \) \* ${lbc_freq}`
		export this_time=`${NECESSAY_FILE_DIR}/da_advance_time.exe ${STARTtime} ${dt} `
		this_time_w=`${NECESSAY_FILE_DIR}/da_advance_time.exe ${STARTtime} ${dt} -w`
		next_time=`${NECESSAY_FILE_DIR}/da_advance_time.exe ${STARTtime} ${dt_plus_one} ` #format : 2018090800

		export member="mem"`printf %03i ${imem}`
		mkdir -p ${ICBC_DIR}/${member}/IC 
		mkdir -p ${ICBC_DIR}/${member}/BC
		
		cd ${ICBC_DIR}/${member}/IC
		export current_work_dir=${ICBC_DIR}/${member}/IC
		ln -sf ${WRFDA_CV3_DIR}/be.dat.cv3 ${ICBC_DIR}/${member}/IC/be.dat
		#ln -sf ${NECESSAY_FILE_DIR}/wrfout_d01_${start_time_wrf} ${ICBC_DIR}/${member}/IC/fg
		ln -sf ${NECESSAY_FILE_DIR}/wrfinput_d01_${this_time} ${ICBC_DIR}/${member}/IC/fg
		ln -sf ${NECESSAY_FILE_DIR}/gribmap.txt ${ICBC_DIR}/${member}/IC/
		ln -sf ${NECESSAY_FILE_DIR}/LANDUSE.TBL ${ICBC_DIR}/${member}/IC/


		sed -i "s/seed_array1.*/seed_array1 = ${this_time},/g" ${NECESSAY_FILE_DIR}/namelist.WRFDA.VAR #modify random seed
		sed -i "s/seed_array2.*/seed_array2 = ${imem},/g" ${NECESSAY_FILE_DIR}/namelist.WRFDA.VAR #modify random seed
		sed -i "s/analysis_date.*/analysis_date = ${this_time_w},/g" ${NECESSAY_FILE_DIR}/namelist.WRFDA.VAR #modify analysis date

		ln -sf ${NECESSAY_FILE_DIR}/namelist.WRFDA.VAR ${ICBC_DIR}/${member}/IC/namelist.input

		ln -sf ${WRFDA_DIR}/da_wrfvar.exe ${ICBC_DIR}/${member}/IC/ #link da_wrfvar.exe
		mpirun ./da_wrfvar.exe
		mv rsl.out.0000 rsl.out
		mv rsl.error.0000 rsl.error
		rm *.0*
		mv ${ICBC_DIR}/${member}/IC/wrfvar_output ${ICBC_DIR}/${member}/IC/wrfinput_${this_time}_${member}

        # ===================#perturbation mask==============================================
		# ncdiff -O ${ICBC_DIR}/${member}/IC/wrfinput_${this_time}_${member} ${ICBC_DIR}/${member}/IC/fg ${ICBC_DIR}/${member}/IC/diff
		# cp ${ICBC_DIR}/${member}/IC/wrfinput_${this_time}_${member} ${ICBC_DIR}/${member}/IC/wrfinput_${this_time}_${member}_tapperd
		# ${python_bin}/python ${NECESSAY_FILE_DIR}/pert_smoother.py
		#=====================================================================================
		it=`expr ${it} \+ 1`	
	done

	
	################################################
	# 2.2 second, BCs
	################################################
	cd ${ICBC_DIR}/${member}/BC
	it=1
	rm ./wrfbdy_d01_${member}_*
	while [[ ${it} -lt ${Nt} ]];do
		dt=`expr \( ${it} \- 1 \) \* ${lbc_freq}`
		dt_plus_one=`expr \( ${it} \) \* ${lbc_freq}`
		# this_time=`${NECESSAY_FILE_DIR}/da_advance_time.exe ${STARTtime} ${dt} -w`
		# next_time=`${NECESSAY_FILE_DIR}/da_advance_time.exe ${STARTtime} ${dt_plus_one} -w` #format : 2018-09-08_00:00:00

		this_time=`${NECESSAY_FILE_DIR}/da_advance_time.exe ${STARTtime} ${dt} `
		next_time=`${NECESSAY_FILE_DIR}/da_advance_time.exe ${STARTtime} ${dt_plus_one} ` #format : 2018090800
		ncks -F -O -d Time,${it} ${NECESSAY_FILE_DIR}/wrfbdy_d01_trfollow ${ICBC_DIR}/${member}/BC/wrfbdy_d01_${member}_${this_time} #seperate LBC for each time
		

		ln -sf ${DART_DIR}/input.nml ${ICBC_DIR}/${member}/BC
		ln -sf ${DART_DIR}/pert_wrf_bc ${ICBC_DIR}/${member}/BC

		cp ${ICBC_DIR}/${member}/BC/wrfbdy_d01_${member}_${this_time} ${ICBC_DIR}/${member}/BC/wrfbdy_this
		ln -sf ${ICBC_DIR}/${member}/IC/wrfinput_${this_time}_${member} ${ICBC_DIR}/${member}/BC/wrfinput_this  #pertubed ICs
		# ln -sf ${NECESSAY_FILE_DIR}/wrfinput_d01_${next_time} ${ICBC_DIR}/${member}/BC/wrfinput_next  #fkc.2025.0927: it seems the link is wrong... should be ICBC_DIR
		ln -sf ${ICBC_DIR}/${member}/IC/wrfinput_${next_time}_${member} ${ICBC_DIR}/${member}/BC/wrfinput_next
		
		./pert_wrf_bc
		mv wrfbdy_this wrfbdy_d01_${member}_${this_time}

		it=`expr ${it} \+ 1`
	done
	ncrcat -O ./wrfbdy_d01_${member}_* ./wrfbdy_d01_${member}
	imem=`expr ${imem} \+ 1`
done
