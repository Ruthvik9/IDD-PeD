#!/bin/bash

# Directory containing subdirectories
root_dir="videos/gopro"

# Iterate over each subdirectory in the root directory
for dir in "$root_dir"/gp_set_*/; do
    echo "Processing directory: $dir"
    
    # Iterate over each .MP4 file in the subdirectory
    for filepath in "$dir"*.MP4; do
        # Check if the file exists
        if [[ -f "$filepath" ]]; then
            echo "Processing $filepath"

            # Temporarily store the stripped version
            temp_output="${filepath%.MP4}_temp.MP4"

            # Run ffmpeg to strip the audio
            ffmpeg -i "$filepath" -c copy -an "$temp_output"

            # Replace the original file with the stripped version
            mv "$temp_output" "$filepath"
        fi
    done
done

echo "Processing complete."
