#!/bin/bash

#export INTELROOT=${HOME}/opt/intel/compilers_and_libraries_2019/linux
#export INTELROOT=/dassw/intel/compilers_and_libraries_2019/linux
export INTELROOT=${HOME}/opt/intel_oneAPI/inteloneapi/compiler/latest/linux/compiler
export LD_LIBRARY_PATH=${INTELROOT}/lib/intel64:${LD_LIBRARY_PATH}

$@
