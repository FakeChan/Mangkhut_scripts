#!/bin/sh
# Script to run the example_fwd.F90 example program
#
# The result is compared to a reference file.
# 
# This script runs only ONE test for NOAA-16 AVHRR

# Set BIN directory if supplied
BIN=$(perl -e 'for(@ARGV){m/^BIN=(\S+)$/o && print "$1";}' $*)
if [ "x$BIN" = "x" ]
then
  BIN=bin
fi

######## Edit this section for your test case input and pathnames ######

# Test case input data
COEF_FILENAME="rtcoef_noaa_18_amsua.dat"
PROF_FILENAME="prof09_18_00.dat"
NPROF=676
NLEVELS=56
DO_SOLAR=0                                # 0 = solar off / 1 = solar on

NCHAN=6
CHAN_LIST=$(seq -s ' ' $NCHAN)            # Space-separated channel-list

NTHREADS=1                                # Number of threads to use (compile RTTOV with OpenMP to exploit this)

CHECK_REF=1                               # Set to 0 to omit check against test reference


# The paths below do not need editing unless you move input files to different locations
# or you require a coefficient file from a different directory

# Path relative to the rttov_test directory:
#TEST_DIR=./test_example_fwd.1
TEST_DIR=${call_rttov_dir}/call_rttov_test
# TEST_DIR=./call_rttov_test
# Paths relative to the rttov_test/${TEST_DIR} directory:
BIN_DIR=${rttov_dir}/bin                           # BIN directory (may be set with BIN= argument)
#REF_TEST_DIR=../test_example_fwd.2           # Test reference data
#COEF_DIR=../../rtcoef_rttov11/rttov7pred101L  # Coefficients directory
COEF_DIR=${rtcoef_dir}
# COEF_DIR=/share/home/lililei1/kcfu/rttov/rtcoef_rttov11/rttov7pred54L
########################################################################

ARG_ARCH=$(perl -e 'for(@ARGV){m/^ARCH=(\S+)$/o && print "$1";}' $*)
if [ ! "x$ARG_ARCH" = "x" ]; then
  ARCH=$ARG_ARCH
fi
if [ "x$ARCH" = "x" ];
then
  echo 'Please supply ARCH'
  exit 1
fi

CWD=$(pwd)
cd $TEST_DIR

echo " "
echo " "
echo " Test forward "
echo " "

echo  "Coef filename:      ${COEF_FILENAME}"
echo  "Input profile file: ${PROF_FILENAME}"
echo  "Number of profiles: ${NPROF}"
echo  "Number of levels:   ${NLEVELS}"
echo  "Do solar:           ${DO_SOLAR}"
echo  "Number of channels: ${NCHAN}"
echo  "Channel list:       ${CHAN_LIST}"
echo  "Number of threads:  ${NTHREADS}"


# Coefficient file
rm -f $COEF_FILENAME
if [ -s $COEF_DIR/$COEF_FILENAME ]; then
  ln -s $COEF_DIR/$COEF_FILENAME
else
  echo "Coef file $COEF_DIR/$COEF_FILENAME not found, aborting..."
  exit 1
fi

$BIN_DIR/example_fwd.exe << EOF
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
echo "Output is in the file ${TEST_DIR}/${PROF_FILENAME}.${ARCH}"

if [ $CHECK_REF -ne 0 ]; then

  if [ -f ${REF_TEST_DIR}/${OUT_FILE} ]; then

    diff -biw ${OUT_FILE}.${ARCH} ${REF_TEST_DIR}/${OUT_FILE} > $DIFF_FILE

    if [ -s $DIFF_FILE ]; then
      echo "--- Diff file contents: ---"
      cat $DIFF_FILE
      echo "---------------------------"
    else
      echo " "
      echo "Diff file has zero size: TEST SUCCESSFUL"
      echo " "
    fi
  else
    echo "Test reference output not found"
  fi
  echo
fi

rm -f $COEF_FILENAME

cd $CWD

exit
