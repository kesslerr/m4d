#!/bin/bash

#experiments=('ERN' 'LRP' 'MMN' 'N170'  'N2pc' 'N400' 'P3') #  
#subjects=('sub-021' 'sub-022' 'sub-023' 'sub-024' 'sub-025' 'sub-026' 'sub-027' 'sub-028' 'sub-029' 'sub-030' 'sub-031' 'sub-032' 'sub-033' 'sub-034' 'sub-035' 'sub-036' 'sub-037' 'sub-038' 'sub-039' 'sub-040')
# DONE 'sub-001' 'sub-002' 'sub-003' 'sub-004' 'sub-005' 'sub-006' 'sub-007' 'sub-008' 'sub-009' 'sub-010' 'sub-011' 'sub-012' 'sub-013' 'sub-014' 'sub-015' 'sub-016' 'sub-017' 'sub-018' 'sub-019' 'sub-020'
# TODO  'sub-021' 'sub-022' 'sub-023' 'sub-024' 'sub-025' 'sub-026' 'sub-027' 'sub-028' 'sub-029' 'sub-030' 'sub-031' 'sub-032' 'sub-033' 'sub-034' 'sub-035' 'sub-036' 'sub-037' 'sub-038' 'sub-039' 'sub-040'

experiments=('MIPDB')
# Extract the IDs using awk and store them in a Bash array
csv_file="data/mipdb/participants.csv"
subjects=($(awk -F',' 'NR>1 {print $1}' "$csv_file"))


# run the multiverse for all experiments and subjects
for experiment in "${experiments[@]}"; do
    for subject in "${subjects[@]}"; do
        echo "Sending $experiment $subject EEGNet to SLURM."
        sbatch src/run-eegnet.slurm $experiment $subject
    done
done
