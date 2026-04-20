#!/bin/sh
#BSUB -q serial
#BSUB -n 24
#BSUB -oo pert.out
#BSUB -eo pert.error
#BSUB -J pert_sub



# for ((i=1;i<=80;i++));do
# 	storedir=`printf %03i $i`
# 	mkdir -p $storedir
# 	ln -sf /share/home/lililei1/kcfu/tc_lekima/advance_ensemble/$i/wrfout_d01_2019-08-06_06:00:00 ./$storedir/
# done

for ((i=1;i<=80;i++));do
	mem=`printf %03i $i`
	pertmem=pert_d01_080606_$mem
	ncdiff ./wrfmean_d01_2019-08-06_06:00:00 ./$mem/wrfout_d01_2019-08-06_06:00:00 $pertmem
done
