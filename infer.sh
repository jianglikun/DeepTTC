#!/bin/bash

# Launches model for inferencing from within the container
# TODO: define what args can be passed

DRUG_DATA=$1
RNA_DATA=$2
OUTPUT_FILE=$3


infer.py $DRUG_DATA $RNA_DATA $OUTPUT_FILE
