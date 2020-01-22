#!/bin/bash

cache="cache-references,"
cache+="cache-misses,"
cache+="L1-dcache-loads,L1-dcache-load-misses,"
cache+="L1-dcache-stores,"
cache+="LLC-loads,LLC-load-misses,"          # number of loads that miss the LLC"
cache+="LLC-stores,LLC-store-misses"         # number of stores that miss the LLC"

simd="fp_arith_inst_retired.128b_packed_double,"
simd+="fp_arith_inst_retired.128b_packed_single,"
simd+="fp_arith_inst_retired.256b_packed_double,"
simd+="fp_arith_inst_retired.256b_packed_single,"
simd+="fp_arith_inst_retired.scalar_double,"
simd+="fp_arith_inst_retired.scalar_single"

perf_5.4 stat -e "$cache,$simd" $@

#perf_5.4 record -e "$cache,$simd" $@
#perf_5.4 report --stdio perf.data

#perf_5.4 annotate
