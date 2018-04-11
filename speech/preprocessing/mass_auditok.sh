#!/bin/bash

wavs=${1%/}
saves=${2%/}

mkdir -p $saves

for wav in $wavs/*.wav
do
  base=$(basename $wav)
  basebase=${base%.wav}
  
  auditok -i $wav -r 8000 -n 0.5 -m 2.0 -s 0.2 -d True > $saves/$basebase.txt
  
  echo file saved at $saves/$basebase.txt
done

