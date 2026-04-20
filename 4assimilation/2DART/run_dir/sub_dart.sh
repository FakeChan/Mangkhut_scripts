#!/bin/sh
#BSUB -J DART_FKC
#BSUB -q largemem
#BSUB -n 120
#BSUB -R "span[ptile=24]"
#BSUB -oo test.out
#BSUB -eo test.err
cd /share/home/lililei1/kcfu/tc_mangkhut/4assimilation/2DART/run_dir
mpirun ./filter
mv post_forward_ope_errors000000 post_forward_ope_errors
mv prior_forward_ope_errors000000 prior_forward_ope_errors 
rm post_forward_ope_errors0*
rm prior_forward_ope_errors0*

touch fkc_dart
