#!/bin/sh
set -eu
#BSUB -J DART_FKC
#BSUB -q largemem
#BSUB -n 120
#BSUB -R "span[ptile=24]"
#BSUB -oo test.out
#BSUB -eo test.err

#------------------------------------------------------------------------------
# Single-observation DART assimilation batch job (LSF).
#
# With `set -eu` the exit status of `mpirun ./filter` propagates to LSF: if
# filter fails, this script stops immediately with the same non-zero status and
# the job is recorded as EXIT (not DONE) - none of the bookkeeping below runs.
#------------------------------------------------------------------------------
cd /share/home/lililei1/kcfu/tc_mangkhut/4assimilation/2DART/run_dir

mpirun ./filter

# Forward-operator error files are only produced when output_forward_op_errors
# is enabled in input.nml.  The renames are guarded so a *successful* run is
# never marked failed merely because these files are absent; a rename that IS
# required still fails the job via `set -e` if it cannot be performed.
if [ -f post_forward_ope_errors000000 ]; then
    mv post_forward_ope_errors000000 post_forward_ope_errors
fi
if [ -f prior_forward_ope_errors000000 ]; then
    mv prior_forward_ope_errors000000 prior_forward_ope_errors
fi
rm -f post_forward_ope_errors0*
rm -f prior_forward_ope_errors0*

# Completion marker: created only when filter and all bookkeeping succeeded.
touch fkc_dart