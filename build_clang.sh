#!/bin/bash

export INTELROOT=${HOME}/opt/intel/compilers_and_libraries_2019/linux

make -f ${1} clean && make -f ${1} all
