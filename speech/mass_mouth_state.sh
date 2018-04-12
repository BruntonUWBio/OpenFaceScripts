#!/bin/bash

vids=${1%/}
txts=${2%/}

for vid in $vids/*.avi
do
  
  filename=${vid##*/}
  basefile=${filename%.avi}
  
  if [ ! -f $txts/$basefile.txt ];
  then
    python3 OpenFaceScripts/speech/speech_recognizer.py -v $vid -t $txts/$basefile.txt
#    rm -r $txts/temp_speech_recognizer
  fi
done
