#!/bin/sh
#=================================================================
#BSUB -J hebing_obs
#BSUB -n 1
#BSUB -o hebing_obs.log
#BSUB -e hebing_obs.err
#=================================================================
matlab -nodesktop -nosplash -nodisplay < hebing_diffchan.m
