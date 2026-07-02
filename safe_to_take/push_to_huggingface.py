from datasets import load_dataset, DatasetDict, Dataset, Audio, concatenate_datasets, Features, Sequence, Value
import pandas as pd
import os
import sys

# audio_dataset = Dataset.from_dict(
#     {
#         "audio": ["audios/patient_1.mp3","audios/patient_2.mp3","audios/patient_3.mp3","audios/patient_4.mp3","audios/patient_5.mp3","audios/patient_6.mp3"],
#         "text": [
#             "REPORT",
#             "REPORT",
#             "REPORT",
#             "REPORT",
#             "REPORT",
#             "REPORT"
#         ]
#         }).cast_column("audio", Audio())

# Patients

hashDict = {}
df = pd.read_excel('recording_data.xlsx', sheet_name='Recording Data')
for index, row in df.iterrows():
    textData = ' '.join(row.astype(str))
    hashDict[index+2] = textData

print(hashDict[2])
print(hashDict[3])

train_audioFiles = []
train_transcriptions = []

test_audioFiles = []
test_transcriptions = []

train_folder_path = './train_wav'  # Replace with the path to your folder
test_folder_path = './test_wav'  # Replace with the path to your folder


# Loop through all files in the specified folder
for filename in os.listdir(train_folder_path):
    # Make sure it's a file (not a directory)
    fullPath = os.path.join(train_folder_path, filename)
    if os.path.isfile(fullPath):
        train_audioFiles.append(fullPath)
        train_transcriptions.append(hashDict[int(filename.split("_")[0])])


# Loop through all files in the specified folder
for filename in os.listdir(test_folder_path):
    # Make sure it's a file (not a directory)
    fullPath = os.path.join(test_folder_path, filename)
    if os.path.isfile(fullPath):
        test_audioFiles.append(fullPath)
        test_transcriptions.append(hashDict[int(filename.split("_")[0])])

train_dataset = Dataset.from_dict(
    {
        "audio": train_audioFiles,
        "sentence": train_transcriptions
        }).cast_column("audio", Audio())

test_dataset = Dataset.from_dict(
    {
        "audio": test_audioFiles,
        "sentence": test_transcriptions
        }).cast_column("audio", Audio())

# audio_dataset[0].setdefault("text", "REPORT")
# audio_dataset[1]["text"] = "REPORT"
# audio_dataset[2]["text"] = "REPORT"
# audio_dataset[3]["text"] = "REPORT"
# audio_dataset[4]["text"] = "REPORT"
# audio_dataset[5]["text"] = "REPORT"

audio_dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})
# audio_dataset = audio_dataset.train_test_split(test_size=0.333, seed=7)
# print(audio_dataset)
# print(audio_dataset['train'][0])
# print(audio_dataset['train'][1])
# print(audio_dataset['train'][2])
# print(audio_dataset['train'][3])
# print(audio_dataset['train'][4])

# print(audio_dataset['test'][0])
# print(audio_dataset['test'][1])
# print(audio_dataset['test'][2])
# print(audio_dataset['test'][3])
# print(audio_dataset['test'][4])

print(audio_dataset)
newDataset = load_dataset("Hani89/medical_asr_recording_dataset")
print(audio_dataset["train"].features)
print(newDataset["train"].features)
sys.exit()

train_dataset = train_dataset.cast_column(
    "audio",
    {
        "array": Sequence(feature=Sequence(Value("float32"))),
        "path": Value("string"),
        "sampling_rate": Value("int64"),
    }
)
test_dataset = test_dataset.cast_column(
    "audio",
    {
        "array": Sequence(feature=Sequence(Value("float32"))),
        "path": Value("string"),
        "sampling_rate": Value("int64"),
    }
)
# newDataset = newDataset.cast_column("audio", Audio(sampling_rate=16000, mono=True))

superSet = DatasetDict({
    "train": concatenate_datasets([newDataset["train"], train_dataset]),
    "test": concatenate_datasets([newDataset["test"], test_dataset])
})
print(newDataset)

print(len(superSet))
print(len(superSet['train']))
print(len(superSet['test']))
# audio_dataset.push_to_hub("David-Mazi/Hey-Omnia", private=True)
# audio_dataset["train"].push_to_hub("David-Mazi/Hey-Omnia", split="train_ift", private=True)
# audio_dataset["test"].push_to_hub("David-Mazi/Hey-Omnia", split="test_ift", private=True)

