import subprocess
import numpy as np
import os 
ensmem_dir='/scratch/lililei1/kcfu/tc_mangkhut/2ens_free_fcst'
work_dir='/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time'
memlist=list(np.arange(1,51))
max_dom=int(os.environ.get('max_dom'))
domain_list=domain_list = [f'd0{i}' for i in range(1, max_dom + 1)]
day=os.environ.get('current_day')
hour=os.environ.get('current_hour')
minute=os.environ.get('current_min')
time=f'{day}_{hour}_{minute}'
# link_target='/share/home/lililei1/kcfu/tc_mangkhut/5cyclingDA/postAnal/10_00_00_fg' #f'{work_dir}/{day}_{hour}_{minute}'
link_target=f'/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/{time}'
subprocess.run(['mkdir','-p',f'{work_dir}'])
subprocess.run(['mkdir','-p',f'{link_target}'])
for domain in domain_list:
    with open(f"wrfout_list_{domain}.txt", "w", encoding="utf-8") as file:
        for imem,mem in enumerate(memlist):
            ensmem=f'{ensmem_dir}/mem{mem:03d}'
            subprocess.run(['mkdir','-p',f'{link_target}'])
            subprocess.run(['ln','-sf',f'{ensmem}/wrfout_{domain}_2018-09-{day}_{hour}:{minute}:00',f'{link_target}/firstguess_{domain}.mem{imem+1:03d}'])
            
            
            file.write(f"{link_target}/firstguess_{domain}.mem{imem+1:03d}" + "\n")
