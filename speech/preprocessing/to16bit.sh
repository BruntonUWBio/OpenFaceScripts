#!/bin/bash

wavdir=$1

mkdir -p $wavdir/not_s16le
mv $wavdir/*.wav $wavdir/not_s16le

for file in $wavdir/not_s16le/*.wav
do
  ffmpeg -i $file -c:a pcm_s16le $wavdir/$(basename $file)
done


