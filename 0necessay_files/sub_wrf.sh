#!/bin/sh

###BSUB -n 80

#BSUB -n 144
#BSUB -eo test.err
#BSUB -oo test.out
#BSUB -R "span[ptile=24]"
#BSUB -J wrf_fkc
#BSUB -q largemem

###BSUB -q fat_384

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
touch wrf_done                  
