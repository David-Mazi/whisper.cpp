from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
import time

# Replace with the desired model (e.g., "openai/whisper-tiny.en")
model_name = "./whisper-tiny.en-vox"
# model_name = "openai/whisper-tiny.en"

# Load processor and model
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

print("Here")
# Load an audio file (replace "path_to_audio.wav" with your audio file)
start_time = time.perf_counter()

audio_path = "test_audio.wav"
audio, rate = librosa.load(audio_path, sr=16000)  # Whisper expects 16 kHz audio

# Preprocess the audio for the model
inputs = processor.feature_extractor(audio, return_tensors="pt", sampling_rate=rate)

input_features = inputs["input_features"]
attention_mask = torch.ones(input_features.shape, dtype=torch.long)  # All positions are valid since no padding is used

# Generate transcription with increased max_new_tokens
generated_ids = model.generate(
    input_features,
    attention_mask=attention_mask,
    max_new_tokens=446,          # Increased from default
    num_beams=5,                 # Using beam search for better quality
    temperature=0.0,             # Lower temperature for more focused sampling
    no_repeat_ngram_size=3,      # Avoid repeating same phrases
    length_penalty=1.0,          # Don't penalize longer outputs
    max_length=500  # Increase as needed
)

# Decode the transcription
transcription = processor.batch_decode(
    generated_ids, 
    skip_special_tokens=True
)[0]

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
print(transcription)
# # Generate transcription
# with torch.no_grad():
#     predicted_ids = model.generate(input_features, attention_mask=attention_mask)

# # Decode the transcription
# transcription = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
# print("Transcription:", transcription)
