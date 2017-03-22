#!/bin/bash
#
# Launches a full computation in an appropriate subdirectory.

if [ -z "$1" ]; then
    echo "Error!"
    echo "Please provide a string that describes the experiment."
    echo "Abort."
    exit 1
fi

dirname=results-`date -u +"%Y-%m-%d-%H.%M"`-$1
mkdir $dirname
cd $dirname
screen -L -d -m ../full-cylindrical
cd ..
