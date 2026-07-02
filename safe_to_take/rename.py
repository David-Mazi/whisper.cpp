import os
import re

def rename_files_in_directory(directory_path):
    # Get a list of all files in the directory
    files = os.listdir(directory_path)
    
    # Dictionary to track the count of files with the same number
    row_count = {}

    # Loop over all files in the directory
    for filename in files:
        # Match files like "Copy of row{number}.m4a", "Copy of row{number}(n).m4a"
        match = re.match(r"Copy of (row|Row)(\d+)(\(\d+\))?\.m4a", filename)
        if match:
            row_number = match.group(2)  # Extract the row number from the filename
            
            # Check if the row_number already has files renamed
            if row_number not in row_count:
                row_count[row_number] = 0

            # Increment count for duplicate files of the same row number
            row_count[row_number] += 1

            # Generate the new filename
            if row_count[row_number] == 1:
                new_filename = f"{row_number}_row.m4a"
            else:
                new_filename = f"{row_number}_row({row_count[row_number] - 1}).m4a"

            # Create the full path to the old and new files
            old_file_path = os.path.join(directory_path, filename)
            new_file_path = os.path.join(directory_path, new_filename)

            # Rename the file in place
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")

# Set the directory path where your files are located
directory_path = './test'  # Change this to the path of your folder

# Run the function to rename the files
rename_files_in_directory(directory_path)
