#!/bin/bash

# Directory containing the .m4a files
input_folder="./david2"
output_folder="./david2_wav"

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"

# Loop through all .m4a files in the input folder
for file in "$input_folder"/*.m4a; do
  # Extract the base filename without extension
  base_name=$(basename "$file" .m4a)

  # Convert the .m4a file to .wav and save it in the output folder
  ffmpeg -i "$file" -ar 16000 -ac 1 "$output_folder/$base_name.wav"
done