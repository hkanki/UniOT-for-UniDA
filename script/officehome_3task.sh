#!/bin/bash
#SBATCH -J officehome
#SBATCH -p ShangHAI
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=all
#SBATCH --mail-user=YOU@MAIL.COM
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
#SBATCH --time 2-00:00

cd .. || exit 1

py_main='main'
gpu=0
dataset='officehome'
exp=${dataset}

sources=(RealWorld Product Art)
targets=(Art RealWorld Clipart)

for i in "${!sources[@]}"
do
    source="${sources[$i]}"
    target="${targets[$i]}"

    echo "Running ${source} -> ${target}"

    python3 "${py_main}.py" \
        --gpu_index "${gpu}" \
        --exp "${exp}" \
        --dataset "${dataset}" \
        --source "${source}" \
        --target "${target}"

    if [ $? -ne 0 ]; then
        echo "Failed: ${source} -> ${target}"
        exit 1
    fi
done

echo "All three transfer tasks finished."