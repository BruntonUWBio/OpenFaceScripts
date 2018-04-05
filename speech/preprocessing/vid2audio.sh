#!/bin/bash

viddir=$1
wavdir=$2

mkdir -p $wavdir

for file in $viddir/*.avi
do
  base=$(basename $file)
  noext=${base%.avi}
  ffmpeg -i $file -vn -acodec copy $wavdir/$noext.wav
done

