#!/bin/bash

scriptdir=$(dirname "$0")
viddir=$1
wavdir=$2

$scriptdir/vid2audio.sh $1 $2
$scriptdir/to16bit.sh $2 
$scriptdir/denoise_script.sh $2


