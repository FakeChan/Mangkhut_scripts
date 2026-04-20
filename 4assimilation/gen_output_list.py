import subprocess
import numpy as np
import os
memlist=list(np.arange(1,51))
max_dom=int(os.environ.get('max_dom'))
base_dir=os.environ.get('base_dir')

for domain in range(1,max_dom+1):
    str_dom=str(domain)
    with open(f"{base_dir}/4assimilation/2DART/run_dir/output_list_{str_dom}.txt", "w", encoding="utf-8") as file:
        for imem,mem in enumerate(memlist):
            file.write(f"output_d0{str_dom}.mem{imem+1:03d}" + "\n")
