#!/bin/bash
experiments=('ERN' 'LRP' 'MMN' 'N170' 'N2pc' 'N400' 'P3')
for experiment in "${experiments[@]}"; do
        
    # Define the directories to process
    directories=(
        #"/ptmp/kroma/m4d/data/processed/"
        #"/ptmp/kroma/m4d/models/eegnet/"
        #"/u/kroma/m4d/models/sliding/"
        "/ptmp/kroma/m4d/data/processed/$experiment/"
        "/u/kroma/m4d/models/sliding/$experiment/"
        "/ptmp/kroma/m4d/models/eegnet/$experiment/"
    )

    # Function to count files in each subfolder
    count_files_in_subfolders() {
        local dir=$1
        echo "Processing directory: $dir"
        for subfolder in "$dir"*/; do
            if [ -d "$subfolder" ]; then
                num_files=$(find "$subfolder" -type f | wc -l)
                echo "Subfolder: $subfolder - Number of files: $num_files"
            fi
        done
    }

    # Process each directory
    for dir in "${directories[@]}"; do
        count_files_in_subfolders "$dir"
    done

done
