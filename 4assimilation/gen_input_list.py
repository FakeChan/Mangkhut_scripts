import subprocess
import numpy as np

ensmem_dir='/scratch/lililei1/kcfu/tc_mangkhut/2ens_free_fcst'
work_dir='/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time'
memlist=list(np.arange(1,51))
domain='d01'
day='10'
hour='00'
minute='00'
# link_target='/share/home/lililei1/kcfu/tc_mangkhut/5cyclingDA/postAnal/10_00_00_fg' #f'{work_dir}/{day}_{hour}_{minute}'
link_target='/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/10_00_00'
subprocess.run(['mkdir','-p',f'{work_dir}'])
subprocess.run(['mkdir','-p',f'{link_target}'])
with open(f"{link_target}/input_list_{domain}.txt", "w", encoding="utf-8") as file:
    for imem,mem in enumerate(memlist):
        ensmem=f'{ensmem_dir}/mem{mem:03d}'
        # subprocess.run(['mkdir','-p',f'{link_target}'])
        # subprocess.run(['ln','-sf',f'{ensmem}/wrfout_{domain}_2018-09-{day}_{hour}:{minute}:00',f'{link_target}/firstguess_{domain}.mem{imem+1:03d}'])
        
        
        file.write(f"{link_target}/firstguess_{domain}.mem{imem+1:03d}" + "\n")
