#!/bin/bash -c

# Variability in Autoreject seed: test with different seeds, preprocessing and EEGNet

# PREPROCESSING

subjects=("sub-001" "sub-002" "sub-003" "sub-004" "sub-005" "sub-006" "sub-007" "sub-008" "sub-009" "sub-010" "sub-011" "sub-012" "sub-013" "sub-014" "sub-015" "sub-016" "sub-017" "sub-018" "sub-019" "sub-020" "sub-021" "sub-022" "sub-023" "sub-024" "sub-025" "sub-026" "sub-027" "sub-028" "sub-029" "sub-030" "sub-031" "sub-032" "sub-033" "sub-034" "sub-035" "sub-036" "sub-037" "sub-038" "sub-039" "sub-040")  
#subjects=("sub-001")  
for subject in ${subjects[@]}; do
    sbatch --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=72 \
        --time=0:10:00 \
        --mem=256G \
        --job-name=ar_preproc_${subject} \
        --output=/u/kroma/m4d/logs/%j_ar_preproc_${subject}.out \
        --error=/u/kroma/m4d/logs/%j_ar_preproc_${subject}.out \
        --mail-user=kessler@cbs.mpg.de \
        --mail-type=ALL \
        --wrap="export OMP_NUM_THREADS=1 && source activate m4d && cd /u/kroma/m4d && python3 /u/kroma/m4d/src/autoreject_var/ar_preproc.py $subject"
done

# EEGNET

subjects=("sub-001" "sub-002" "sub-003" "sub-004" "sub-005" "sub-006" "sub-007" "sub-008" "sub-009" "sub-010" "sub-011" "sub-012" "sub-013" "sub-014" "sub-015" "sub-016" "sub-017" "sub-018" "sub-019" "sub-020" "sub-021" "sub-022" "sub-023" "sub-024" "sub-025" "sub-026" "sub-027" "sub-028" "sub-029" "sub-030" "sub-031" "sub-032" "sub-033" "sub-034" "sub-035" "sub-036" "sub-037" "sub-038" "sub-039" "sub-040")  
#subjects=("sub-001")  
for subject in ${subjects[@]}; do
    sbatch --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=72 \
        --time=0:15:00 \
        --mem=256G \
        --job-name=ar_en_${subject} \
        --output=/u/kroma/m4d/logs/%j_ar_en_${subject}.out \
        --error=/u/kroma/m4d/logs/%j_ar_en_${subject}.out \
        --mail-user=kessler@cbs.mpg.de \
        --mail-type=ALL \
        --wrap="export OMP_NUM_THREADS=1 && source activate m4d && cd /u/kroma/m4d && python3 /u/kroma/m4d/src/autoreject_var/ar_eegnet.py $subject"
done

# ANALYSIS

python3 /u/kroma/m4d/src/autoreject_var/ar_analysis.py