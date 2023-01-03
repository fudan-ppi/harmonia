#!/bin/bash
# This shell is used to trigger for compile.
# After succssfully compiled, there will be three executable file: bpt_test, generate-partially-sort-dataset and regular_bpt_test
sh clean.sh

# cmake -DCMAKE_BUILD_TYPE=Release -DORDER=32 -DNARROW=16 .
cmake -DCMAKE_BUILD_TYPE=Release -DORDER=64 -DNARROW=8 .
#cmake -DCMAKE_BUILD_TYPE=Debug -DORDER=32 -DNARROW=16 .
# cmake -DCMAKE_BUILD_TYPE=Debug -DORDER=64 -DNARROW=8 .

make -j
