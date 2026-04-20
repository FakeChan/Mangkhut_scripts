import subprocess
import numpy as np
import os 
ensmem_dir=os.environ.get('ensmem_dir')
max_dom=int(os.environ.get('max_dom'))
memlist=list(np.arange(1,51))
day=os.environ.get('current_day')
hour=os.environ.get('current_hour')
minute=os.environ.get('current_min')
link_target=f'{ensmem_dir}'
# subprocess.run(['mkdir','-p',f'{work_dir}'])
subprocess.run(['mkdir','-p',f'{link_target}'])
for domain in range(1,max_dom+1):
    str_dom=str(domain)
    with open(f"{link_target}/input_list_d0{str_dom}.txt", "w", encoding="utf-8") as file:
        for imem,mem in enumerate(memlist):
            # ensmem=f'{ensmem_dir}/mem{mem:03d}'
            
            
            file.write(f"{link_target}/firstguess_d0{domain}.mem{imem+1:03d}" + "\n")

    