#!/bin/sh
# Script to run the example_rttovscatt_fwd.F90 example program
#
# The result is compared to a reference file.
# 
# This script runs only ONE test for NOAA-15 AMSU-A

# Set BIN directory if supplied
BIN=$(perl -e 'for(@ARGV){m/^BIN=(\S+)$/o && print "$1";}' $*)
if [ "x$BIN" = "x" ]
then
  BIN=bin
fi

######## Edit this section for your test case input and pathnames ######

# Test case input data
COEF_FILENAME="rtcoef_noaa_18_amsua.dat"
MIETABLE_FILENAME="mietable_noaa_amsua.dat" # Location of this file is set below in $MIETABLE_DIR
PROF_FILENAME="prof10_00_00.dat"
NPROF=676
NLEVELS=56

# The TOTALICE and SNOWRAIN_UNIT options must match the input data in prof.dat
TOTALICE=${use_total_ice}                                  # 0 = separate ciw, snow; 1 = use totalice
SNOWRAIN_UNIT=1                             # units for snow, rain: 0 = kg/m2/s; 1 = kg/kg

NCHAN=15
CHAN_LIST=$(seq -s ' ' $NCHAN)              # Space-separated channel-list
NTHREADS=1
CHECK_REF=0                                # Set to 0 to omit check against test reference


# The paths below do not need editing unless you move input files to different locations
# or you require a coefficient file from a different directory

# Path relative to the rttov_test directory:
TEST_DIR=${call_rttov_dir}/call_rttov_test

# Paths relative to the rttov_test/${TEST_DIR} directory:
BIN_DIR=${rttov_dir}/bin    # BIN directory (may be set with BIN= argument)
REF_TEST_DIR=../test_example_rttovscatt_fwd.2  # Test reference data
COEF_DIR=${rtcoef_dir}    # Coefficients directory
MIETABLE_DIR=/share/home/lililei1/kcfu/rttov/rtcoef_rttov11/mietable    # Mietables directory
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
echo " Test RTTOV-SCATT forward "
echo " "

echo  "Coef filename:            ${COEF_FILENAME}"
echo  "Mietable filename:        ${MIETABLE_FILENAME}"
echo  "Input profile file:       ${PROF_FILENAME}"
echo  "Number of profiles:       ${NPROF}"
echo  "Number of levels:         ${NLEVELS}"
echo  "Totalice:                 ${TOTALICE}"
echo  "Snow/rain unit:           ${SNOWRAIN_UNIT}"
echo  "Number of channels:       ${NCHAN}"
echo  "Channel list:             ${CHAN_LIST}"
echo  "Number of threads:        ${NTHREADS}"

# Coefficient files
COEF_FILENAME="rtcoef_noaa_18_amsua.dat"
if [ ! -f $COEF_FILENAME ]; then
  echo "Coef file $COEF_FILENAME not found, aborting..."
  exit 1
fi

MIETABLE_FILENAME=$MIETABLE_DIR/$MIETABLE_FILENAME
if [ ! -f $MIETABLE_FILENAME ]; then
  echo "Coef file $MIETABLE_FILENAME not found, aborting..."
  exit 1
fi

$BIN_DIR/example_rttovscatt_fwd.exe << EOF
"${COEF_FILENAME}"    , Coefficient filename
"${MIETABLE_FILENAME}", Mietable filename
${PROF_FILENAME}      , Input profile filename
${NPROF}              , Number of profiles
${NLEVELS}            , Number of levels
${TOTALICE}           , Set totalice true/false
${SNOWRAIN_UNIT}      , Set units for snow/rain
${NCHAN}              , Number of channels
${CHAN_LIST}          , Channel numbers
${NTHREADS}           ,   Number of threads
EOF

if [ $? -ne 0 ]; then
  echo " "
  echo "TEST FAILED"
  echo " "
  exit 1
fi

if [ $CHECK_REF -ne 0 ]; then

  OUT_FILE=output_example_rttovscatt_fwd.dat
  DIFF_FILE=diff_example_rttovscatt_fwd.${ARCH}

  mv ${OUT_FILE} ${OUT_FILE}.${ARCH}

  if [ $? -ne 0 ]; then
    echo "Expected output file not found"
    exit 1
  fi

  echo
  echo "Output is in the file ${TEST_DIR}/${OUT_FILE}.${ARCH}"

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

fi

echo
cd $CWD

exit
