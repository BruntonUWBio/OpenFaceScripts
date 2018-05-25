#!/bin/bash

auds=${1%/}
txts=${2%/}

for aud in $auds/*.wav
do

  filename=${wav##*/}
  basefile=${filename%.wav}

  if [ ! -f $txts/$basefile.txt ];
  then
    python3 OpenFaceScripts/speech/audio_features.py -a $aud -od $txts
  fi
done

