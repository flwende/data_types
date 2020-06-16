#!/bin/bash

filename_prefix="data_${1}"
command=${@:2}

factors=(
    0.0
    0.000001
    0.000005
    0.00001
    0.00005    
    0.0001
    0.0005
    0.001
    0.005
    0.01
    0.05
    0.1
    0.2
    0.3
    0.4
    0.5
    0.6
    0.7
    0.8
    0.9
    1.0
)

#for value in ${factors[@]}
#do
#    #filename="${filename_prefix}_diffusion${value}.txt"
#    filename="${filename_prefix}.txt"
#    echo "diffusion ${value}"
#    export DIFFUSION_FACTOR=${value}
#    #for i in `seq 1 1 10`
#    for i in `seq 1 1 1`
#    do
#	${command} | grep "^[^#]" >> ${filename}
#    done
#    echo -e "\n" >> ${filename}
#done


#extents=(
#    "1024"
#    "1048576"
#    "16777216"
#    "32 32"
#    "64 16"
#    "128 8"
#    "256 4"
#    "1024 1024"
#    "4096 256"
#    "16384 64"
#    "65536 16"
#    "262144 4"
#    "16384 1024"
#    "65536 256"
#    "262144 64"
#    "32 8 4"
#    "64 4 4"
#    "128 4 2"
#    "256 2 2"
#    "256 64 64"
#    "512 32 64"
#    "1024 32 32"
#    "256 256 256"
#    "1024 128 128"
#    "4096 64 64"
#    "16384 32 32"
#    "65536 16 16"
#    "262144 8 8"
#)
extents=(
    "32768"
    "16777216"
    "31 1024"
    "32 1024"
    "33 1024"
    "1023 32"
    "1024 32"
    "1025 32"
    "16383 1024"
    "16384 1024"
    "16385 1024"
    "31 32 32"
    "32 32 32"
    "33 32 32"
    "1023 8 4"
    "1024 8 4"
    "1025 8 4"
    "16383 32 32"
    "16384 32 32"
    "16385 32 32"
)

for value in "${extents[@]}"
do
    filename="${filename_prefix}.txt"
    
    echo "# extent: ${value}" >> ${filename}
    echo "# extent: ${value}"
    for i in `seq 1 1 10`
    do
	${command} ${value} | grep "^[^#]" >> ${filename}
    done
    echo -e "\n" >> ${filename}
done
