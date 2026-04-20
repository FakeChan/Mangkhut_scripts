#!/bin/sh
#BSUB -n 144
#BSUB -R "span[ptile=24]"
#BSUB -q mpi
# 注意下面使用了 %I，LSF 会自动将输出日志按成员标号分开
#BSUB -eo test_%I.err
#BSUB -oo test_%I.out

# 获取 LSF 自动分配的数组索引 (1 到 50)
imem=$LSB_JOBINDEX
mem=$(printf "%03d" $imem)

# base_dir 环境变量会自动从驱动脚本继承过来
cycle_run_wrf_dir=${base_dir}/5cyclingDA/run_wrf/${current_time}/${mem}

# 进入专属成员目录
cd ${cycle_run_wrf_dir} || exit 1

export I_MPI_DEVICE=rdssm
ulimit -s unlimited
ulimit -d unlimited
export WRFIO_NCD_LARGE_FILE_SUPPORT=1

mpiexec.hydra ./wrf.exe

mv rsl.error.0000 rsl.err
mv rsl.out.0000 rsl.out

# 清理无用流浪文件，并生成标志位
rm -f rsl.error* rsl.out.*
touch wrf_done_${current_time}
