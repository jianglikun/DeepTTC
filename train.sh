#!/bin/bash

# arg 1 CUDA_VISIBLE_DEVICES
# arg 2 CANDLE_DATA_DIR
# arg 3 CANDLE_CONFIG

CANDLE_MODEL=/DeepTTC/DeepTTC_candle.py
if [[ "$#" < 2  ]] ; then
	    echo "Illegal number of parameters"
	    echo "CUDA_VISIBLE_DEVICES CANDLE_DATA_DIR are required"
	    exit -1
fi

CUDA_VISIBLE_DEVICES=$1; shift
CANDLE_DATA_DIR=$1; shift
CMD="python3 ${CANDLE_MODEL} $@"



echo "using container "
echo "using CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "using CANDLE_DATA_DIR ${CANDLE_DATA_DIR}"
echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"
echo "running command ${CMD}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $CMD
