#!/bin/bash -l

experiments=("ERN" "LRP" "MMN" "N170" "N2pc" "N400" "P3")
for experiment in ${experiments[@]}; do
    sbatch --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=72 \
        --time=1:00:00 \
        --mem=256G \
        --job-name=tr_group_${experiment} \
        --output=/u/kroma/m4d/logs/%j_tr_group_${experiment}.out \
        --error=/u/kroma/m4d/logs/%j_tr_group_${experiment}.out \
        --mail-user=kessler@cbs.mpg.de \
        --mail-type=ALL \
        --wrap="export OMP_NUM_THREADS=1 && source activate m4d && cd /u/kroma/m4d && python3 /u/kroma/m4d/src/sliding_group_tfce.py $experiment"
done
