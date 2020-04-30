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



for factor in ${factors[@]}
do
    filename="${filename_prefix}_diffusion${factor}.txt"
    export DIFFUSION_FACTOR=${factor}

    echo "diffusion ${factor}"
    ${command} >> "${filename_prefix}.txt"
    
    for i in `seq 1 1 10`
    do
	echo "iteration: ${i}"
	${command} >> ${filename}
    done
done
