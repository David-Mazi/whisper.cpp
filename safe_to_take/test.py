import pandas as pd

hashDict = {}
df = pd.read_excel('recording_data.xlsx', sheet_name='Recording Data')
for index, row in df.iterrows():
    textData = ' '.join(row.astype(str))
    hashDict[index+2] = textData

counter = 1
patients = {}
import os

folder_path = './test_wav'  # Replace with the path to your folder

# Loop through all files in the specified folder
for filename in os.listdir(folder_path):
    # Make sure it's a file (not a directory)
    fullPath = os.path.join(folder_path, filename)
    if os.path.isfile(fullPath):
        patients[f"Test {counter}"] = (fullPath, hashDict[int(filename.split("_")[0])])
        counter = counter + 1
