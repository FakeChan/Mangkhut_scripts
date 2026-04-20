#!/bin/sh
# This script updates a base wrfinput_d01 file with variables
# from the last time step of a previous wrfout file.
# The variable lists are taken directly from the user-provided script.
#
# It expects two files in the current directory:
#   - wrfinput_d01: The stable "base" file to be updated.
#   - previous_wrfout_d01: A symlink to the wrfout file from the previous cycle.

set -e

echo "      -> Running stability update script with user-defined variables..."

# Define the input/output files
BASE_WRFINPUT="wrfinput_d01"
PREV_WRFOUT="previous_wrfout_d01"
TEMP_LAST_STEP_FILE="temp_last_step.nc"

if [ ! -f "$BASE_WRFINPUT" ] || [ ! -L "$PREV_WRFOUT" ]; then
    echo "      ERROR: Missing base wrfinput_d01 or symlink to previous_wrfout_d01."
    exit 1
fi

# Step 1: Extract the last time step from the previous wrfout file.
# This is the critical step to adapt a multi-time-step wrfout file for this purpose.
echo "      -> Extracting last time step from ${PREV_WRFOUT}..."
ncks -O -d Time,-1 ${PREV_WRFOUT} ${TEMP_LAST_STEP_FILE}

# Step 2: Define the variable lists exactly as provided by the user.
common_vars="MU,PSFC,Q2,T2,TH2,TSK,U10,V10,CLDFRA,OM_TMP,OM_U,OM_V,P,PB,PH,QCLOUD,QGRAUP,QICE,QRAIN,QVAPOR,THM,U,V,W"
other_vars="TMOML,OM_ML,TSK,CANWAT,ALBBCK,H0ML,LAI,VEGFRA"

# Step 3: Use the extracted data (temp_last_step.nc) to overwrite the variables
# in the base wrfinput file, mimicking the original script's logic.
echo "      -> Updating common variables in ${BASE_WRFINPUT}..."
ncks -A -v $common_vars ${TEMP_LAST_STEP_FILE} ${BASE_WRFINPUT}

echo "      -> Updating other variables in ${BASE_WRFINPUT}..."
ncks -A -v $other_vars ${TEMP_LAST_STEP_FILE} ${BASE_WRFINPUT}

# Step 4: Clean up the temporary file.
rm ${TEMP_LAST_STEP_FILE}

echo "      -> Update complete. The wrfinput_d01 is ready for integration."