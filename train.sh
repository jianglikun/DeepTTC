#!/bin/bash

#########################################################################
### THIS IS A TEMPLATE FILE. SUBSTITUTE #PATH# WITH THE MODEL EXECUTABLE.
#########################################################################


# arg 1 CUDA_VISIBLE_DEVICES
# arg 2 CANDLE_DATA_DIR
# arg 3 CANDLE_CONFIG

### Path to your CANDLEized model's main Python script###
CANDLE_MODEL=/DeepTTC/DeepTTC_candle.py

if [ $# -lt 2 ] ; then
	echo "Illegalnumber of paramaters"
	echo "Illegal number of parameters"
	echo "CUDA_VISIBLE_DEVICES CANDLE_DATA_DIR are required"
	exit -1
fi

if [ $# -eq 2 ] ; then
	CUDA_VISIBLE_DEVICES=$1 ; shift
	CANDLE_DATA_DIR=$1 ; shift
	CMD="python ${CANDLE_MODEL}"
	echo "CMD = $CMD"

else
        CUDA_VISIBLE_DEVICES=$1 ; shift
        CANDLE_DATA_DIR=$1 ; shift

	# if third arg is a file, then set --config_file
	if [ -f $1 ] ; then
  		CANDLE_CONFIG=$1 ; shift
  		CMD="python ${CANDLE_MODEL} --config_file $CANDLE_CONFIG $@"
  		echo "CMD = $CMD $@"

	# else don't set --config_file
	else
		CMD="python ${CANDLE_MODEL} $@"
		echo "CMD = $CMD"
	fi
fi

# Display runtime arguments
echo "using CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "using CANDLE_DATA_DIR ${CANDLE_DATA_DIR}"
echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"
echo "running command ${CMD}"

# Set up environmental variables and execute model
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $CMD
