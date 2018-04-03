#!/bin/bash

scriptdir=$(dirname "$0")
wavdir=$1

mkdir -p $wavdir/noisy
mv $wavdir/*.wav $wavdir/noisy

for file in $wavdir/noisy/*.wav
do  
  python3 $scriptdir/denoise_audio.py $file $scriptdir/audio.wav
  sox -r 8000 $scriptdir/audio.wav -r 48000 $scriptdir/audio_up.raw
  $scriptdir/rnnoise/examples/rnnoise_demo $scriptdir/audio_up.raw $scriptdir/audio_denoised.raw
  sox -t s16 -c 1 -r 48000 $scriptdir/audio_denoised.raw -r 8000 $wavdir/$(basename $file)
  rm $scriptdir/audio*
done




