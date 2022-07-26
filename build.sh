#!/bin/bash
RECIPE=$1
BASE_PATH="images"
WRITABLE_PATH="writable"
IMAGE_PATH="images"

# Prepare paths
declare -a paths=($BASE_PATH $WRITABLE_PATH $IMAGE_PATH)
for directory in "${paths[@]}"
do
	if [ ! -d $directory ]
	then
		mkdir $directory
	fi
done

# Extract first element splitted on '.' from recipe file name
# Currently works ONLY for the files in the same directory
oldIFS=$IFS
IFS=.
read -a splitted <<< "$RECIPE"
echo $RECIPE
NAME=${splitted[0]}
IFS=$oldIFS
RECIPE=$(IFS=. ; echo "${splitted[*]}")
echo $RECIPE
DATE=$(date +%Y_%m_%d)
LABELED_NAME=$NAME-$DATE


# Create base sif image from recipe (definition) file
singularity build --fakeroot --sandbox $BASE_PATH/$LABELED_NAME.sif $RECIPE
singularity build --fakeroot --sandbox $WRITABLE_PATH/$LABELED_NAME $BASE_PATH/$LABELED_NAME.sif
