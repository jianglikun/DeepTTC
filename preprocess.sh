#!/bin/bash

COMMUN_PREPROCESS=/DeepTTC
CANDLE_PREPROCESS=$COMMUN_PREPROCESS

if [[ "$#" -ne 1 ]] ; then
	    echo "Illegal number of parameters"
	        echo "CANDLE_DATA_DIR required"
		    exit -1
	    fi

	    CANDLE_DATA_DIR=$1
	    #CANDLE_CONFIG=$2

	    #CMD="python ${CANDLE_PREPROCESS} --config_file ${CANDLE_CONFIG}"
	    CMD="python# ${CANDLE_PREPROCESS}"

	    echo "using CANDLE_DATA_DIR ${CANDLE_DATA_DIR}"
	    echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"
	    echo "running command ${CMD}"

	    CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $CMD
