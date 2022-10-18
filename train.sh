#!/bin/bash

# arg 1 CUDA_VISIBLE_DEVICES
# arg 2 CANDLE_DATA_DIR
# arg 3 CANDLE_CONFIG

CANDLE_MODEL=/DeepTTC/DeepTTC_candle.py
if [[ "$#" -ne 2 && "$#" -ne 3 ]] ; then
	    echo "Illegal number of parameters"
	    echo "CUDA_VISIBLE_DEVICES CANDLE_DATA_DIR are required"
		echo "CANDLE_CONFIG is an optional parameter. It defines a path to the parameter list for the model."
	    exit -1
fi

CUDA_VISIBLE_DEVICES=$1
CANDLE_DATA_DIR=$2
CANDLE_CONFIG=$3

CMD=""

if [ ! -z "${CANDLE_CONFIG}" ]; then
        if [ ! -f "$CANDLE_CONFIG" ]; then
            echo "Cannot read configuration file! If you want to run model with default parameters leave third option empty."
			exit -1
		else
			CMD="python3 ${CANDLE_MODEL} --config_file ${CANDLE_CONFIG}"
        fi
else
	CMD="python3 ${CANDLE_MODEL}"
fi



echo "using container "
echo "using CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "using CANDLE_DATA_DIR ${CANDLE_DATA_DIR}"
echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"
echo "running command ${CMD}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $CMD
