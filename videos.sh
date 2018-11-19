#!/bin/bash

mypath=$1/*
for f in $mypath;
do
	filename=$(basename -- "$f")
	#myPath=$(realpath $f)
	#echo "$myPath $(ls $myPath | wc -l) $2" >> $3
	mkdir -p output/$1/$filename
	ffmpeg -i $mypath$filename -vf fps=3 /home/darius/Downloads/dataset-minim/output/$1/$filename/out%d.png -y
done