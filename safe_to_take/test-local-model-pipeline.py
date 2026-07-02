from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
import numpy as np
import time
import sys
from transformers import pipeline


# Replace with the desired model (e.g., "openai/whisper-tiny.en")
# model_name = "./whisper-tiny.en-vox"
# model_name = "David-Mazi/whisper-tiny-vox"
# model_name = "openai/whisper-tiny.en"
model_name = './whisper-tiny.en-vox'
whisper_asr = pipeline(
    "automatic-speech-recognition", 
    model=model_name,
    chunk_length_s=30,
    generate_kwargs={
        # "max_length": 448,        # Allow for longer outputs
        # "temperature": 0.7,       # Add some randomness
        # "length_penalty": 1.0,    # Neutral penalty for output length
        # "repetition_penalty": 1.2 # Penalize repeated tokens
        "max_new_tokens": 444,          # Increased from default
        "num_beams": 3,                 # Using beam search for better quality
        "temperature": 0.0,             # Lower temperature for more focused sampling
        "no_repeat_ngram_size": 3,      # Avoid repeating same phrases
        "length_penalty": 1.0
    }
)
# model_check_point = './whisper-tiny.en-vox-old/checkpoint-300'

start_time = time.perf_counter()

# Patients
patients = {
    "Test 1": ("audios/patient_1.m4a", "REPORT"),
    "Test 2": ("audios/patient_2.m4a", "REPORT"),
    "Test 3": ("audios/patient_3.m4a", "REPORT"),
    "Test 4": ("audios/patient_4.m4a", "REPORT"),
    "Test 5": ("audios/patient_5.m4a", "REPORT"),
    "Test 6": ("audios/patient_6.m4a", "REPORT")
}
# Counters
total = 0
counter = 0

# Tests
start_time = time.perf_counter()
for testName, testData in patients.items():
    print(testName)
    source_text = testData[1].lower()
    source_set = set(source_text.split())
    source_audio = testData[0]

    audio_path = source_audio
    audio, rate = librosa.load(audio_path, sr=16000)  # Whisper expects 16 kHz audio
    final_transcription = whisper_asr(audio)["text"]

    unsanitize = final_transcription
    unsanitize = unsanitize.lower()
    unsanitize = unsanitize.replace(" slash ", "/")
    unsanitize = unsanitize.replace("-slash-", "/")
    unsanitize = unsanitize.replace(" over ", "/")
    result_text = unsanitize.replace("-over-", "/")
    result_set = set(result_text.split())
    print(source_text)
    print('\n')
    print(result_text)
    for i in result_set:
        if i in source_set:
            counter += 3
        else:
            counter -= 0.0
        total += 3

    for j in source_set:
        if j in result_set:
            counter += 3
        else:
            counter -= 0.0
        total += 3

    print("Running total after "+ testData[0] + ": " + str((counter/total)*100) + "%")
    endOfIter = time.perf_counter()
    print(f"Elapsed time: {(endOfIter - start_time)} seconds")
end_time = time.perf_counter()
print("Final Score: " + str((counter/total)*100) + "%")
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time/len(patients)} seconds")
