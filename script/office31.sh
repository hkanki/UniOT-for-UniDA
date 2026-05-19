#!/bin/bash

set -e

cd ..

py_main='main'
gpu=0

dataset='office31'
domains=(amazon dslr webcam)
exp=${dataset}

mkdir -p output

for source in ${domains[@]}
do
    for target in ${domains[@]}
    do
        if [[ "${source}" != "${target}" ]]
        then
            echo "Running ${source} -> ${target}"
            python3 ${py_main}.py --gpu_index ${gpu} --exp ${exp} --dataset ${dataset} --source ${source} --target ${target}
        fi
    done
done