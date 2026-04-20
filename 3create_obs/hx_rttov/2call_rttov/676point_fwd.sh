#!/bin/sh
# Script to run the EXAMPLE_FWD code
#
# The result is compared to a reference file. See users guide
# to note what are the "normal" differences.
# 
# This script runs only ONE test for NOAA-16 AVHRR

# Set BIN directory if supplied
BIN=`perl -e 'for(@ARGV){m/^BIN=(\S+)$/o && print "$1";}' $*`
if [ "x$BIN" = "x" ]
then
  BIN=bin
fi

######## Edit this section for your pathnames and test case input ######

# Path relative to the rttov_test directory:
#TES_T=/share/home/lililei1/lfzhou/models/rttovsimplify/rttov_test/2call_rttov_d01
TEST=${call_rttov_dir}/call_rttov_test
BINDIR=/share/home/lililei1/kcfu/rttov/bin                        # BIN directory (may be set with BIN= argument)
#REF_TEST=./2call_rttov_d02
#REF_TEST=./2call_rttov_d02

# DATART=/share/home/lililei1/kcfu/rttov/rtcoef_rttov11/rttov7pred101L    # Coefficients directory
DATART=${rtcoef_dir}
echo DATART
# Test case input data
COEF_FILENAME="rtcoef_fy4_1_giirs_local.dat"

PROF_FILENAME="prof10_00_00.dat"
NPROF=676
NLEVELS=56
DO_SOLAR=0      # 0 = solar off / 1 = solar on
NCHAN=1025
CHAN_LIST=1650
CHECK_REF=1
NTHREADS=1
# It is possible to specify input emissivity and BRDF values below.
# Alternatively set them to zero to use RTTOV internal defaults.

########################################################################

ARG_ARCH=`perl -e 'for(@ARGV){m/^ARCH=(\S+)$/o && print "$1";}' $*`
if [ ! "x$ARG_ARCH" = "x" ]; then
  ARCH=$ARG_ARCH
fi
if [ "x$ARCH" = "x" ];
then
  echo 'Please supply ARCH'
  exit 1
fi

CWD=`pwd`
cd $TEST

echo " "
echo " "
echo " Test forward "
echo " "

echo  "Coef filename:      ${COEF_FILENAME}"
echo  "Input profile file: ${PROF_FILENAME}"
echo  "Number of profiles: ${NPROF}"
echo  "Number of levels:   ${NLEVELS}"
echo  "Do solar:           ${DO_SOLAR}"


# Coefficient file
rm -f $COEF_FILENAME
if [ -s $DATART/$COEF_FILENAME ]; then
  ln -s $DATART/$COEF_FILENAME
else
  echo "Coef file $DATART/$COEF_FILENAME not found, aborting..."
  exit 1
fi

$BINDIR/example_fwd.exe << EOF
${COEF_FILENAME}, Coefficient filename
${PROF_FILENAME}, Input profile filename
${NPROF}        , Number of profiles
${NLEVELS}      , Number of levels
${DO_SOLAR}     , Turn solar radiation on/off
${NCHAN}        , Number of channels
${CHAN_LIST}    , Channel numbers
${NTHREADS}     , Number of threads
EOF

if [ $? -ne 0 ]; then
  echo " "
  echo "TEST FAILED"
  echo " "
  exit 1
fi

OUT_FILE=output_example_fwd.dat
DIFF_FILE=diff_example_fwd.${ARCH}

mv ${OUT_FILE} ${PROF_FILENAME}.${ARCH}

if [ $? -ne 0 ]; then
  echo "Expected output file not found"
  exit 1
fi

echo
echo "Output is in the file ${TEST}/${OUT_FILE}.${ARCH}"

#echo "${REF_TEST}/${OUT_FILE}"
#if [ -f ${REF_TEST}/${OUT_FILE} ]; then
#  echo "LILI"

 # diff -biw ${OUT_FILE}.${ARCH} ${REF_TEST}/${OUT_FILE} > $DIFF_FILE

#  if [ `stat -c %s $DIFF_FILE` -eq 0 ]; then
#    echo " "
#    echo "Diff file has zero size: TEST SUCCESSFUL"
#    echo " "
#  else
#    echo "--- Diff file contents: ---"
#    cat $DIFF_FILE
#    echo "---------------------------"
#  fi
#else
#  echo "Test reference output not found"
#fi
#echo

rm -f $COEF_FILENAME

cd $CWD

exit
