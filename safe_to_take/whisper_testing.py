import whisper
import time

modelName = "tiny.en"
model = whisper.load_model(modelName)
# print(whisper.available_models())

# result = model.transcribe("test_audio.wav")
# result = model.transcribe("patient_1.m4a")
# print(result["text"])

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

# Model SetUp
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

# Replace with the desired model (e.g., "openai/whisper-tiny.en")
# model_name = "David-Mazi/whisper-tiny-vox"
model_name = './whisper-tiny.en-vox'
# model_check_point = './whisper-tiny.en-vox/checkpoint-200'

# Load processor and model
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)


# Tests
print(modelName)
start_time = time.perf_counter()
for testName, testData in patients.items():
    print(testName)
    source_text = testData[1]
    source_set = set(source_text.lower().split())
    source_audio = testData[0]

    # unsanitize = model.transcribe(source_audio)["text"]

    # Load an audio file (replace "path_to_audio.wav" with your audio file)
    audio_path = source_audio
    audio, rate = librosa.load(audio_path, sr=16000)  # Whisper expects 16 kHz audio

    # Preprocess the audio for the model
    inputs = processor.feature_extractor(audio, return_tensors="pt", sampling_rate=rate)

    input_features = inputs["input_features"]
    attention_mask = torch.ones(input_features.shape, dtype=torch.long)  # All positions are valid since no padding is used

    # Generate transcription with increased max_new_tokens
    generated_ids = model.generate(
        input_features,
        max_new_tokens=444,          # Increased from default
        num_beams=5,                 # Using beam search for better quality
        temperature=0.0,             # Lower temperature for more focused sampling
        no_repeat_ngram_size=3,      # Avoid repeating same phrases
        length_penalty=1.0,          # Don't penalize longer outputs
    )

    # Decode the transcription
    transcription = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True
    )[0]

    unsanitize = transcription

    unsanitize = unsanitize.replace(" slash ", "/")
    unsanitize = unsanitize.replace("-slash-", "/")
    unsanitize = unsanitize.replace(" over ", "/")
    unsanitize = unsanitize.replace("-over-", "/")
    result_text = unsanitize.replace("patient", "patient pt")
    result_set = set(result_text.lower().split())
    print(source_text)
    print('')
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
end_time = time.perf_counter()
print("Final Score: " + str((counter/total)*100) + "%")
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time/len(patients)} seconds")
