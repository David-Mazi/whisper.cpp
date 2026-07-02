from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
import numpy as np
import time
import sys

# Replace with the desired model (e.g., "openai/whisper-tiny.en")
# model_name = "./whisper-tiny.en-vox"
# model_name = "David-Mazi/whisper-tiny-vox"
# model_name = "openai/whisper-tiny.en"
model_name = './whisper-tiny.en-vox'
# model_check_point = './whisper-tiny.en-vox-old/checkpoint-300'

# Load processor and model
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

print("Here")
# Load an audio file (replace "path_to_audio.wav" with your audio file)
def split_audio(audio, sample_rate, chunk_duration=30):
    """
    Splits an audio signal into smaller chunks.
    
    Args:
    - audio: np.array, the raw audio signal.
    - sample_rate: int, the sample rate of the audio.
    - chunk_duration: int, duration of each chunk in seconds.

    Returns:
    - List of audio chunks.
    """
    chunk_size = sample_rate * chunk_duration  # Number of samples per chunk
    num_chunks = int(np.ceil(len(audio) / chunk_size))
    
    return [audio[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

def split_audio_with_overlap(audio, sample_rate, chunk_duration=30, overlap=2):
    chunk_size = sample_rate * chunk_duration
    overlap_size = sample_rate * overlap
    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + chunk_size, len(audio))
        chunks.append(audio[start:end])
        start += chunk_size - overlap_size
    return chunks

start_time = time.perf_counter()

# Load the audio file
audio_path = "glen.wav"
audio, rate = librosa.load(audio_path, sr=16000)  # Whisper expects 16 kHz audio

# Split into 30-second chunks
audio_chunks = split_audio_with_overlap(audio, rate, chunk_duration=30)

# Preprocess the audio for the model
def transcribe_chunk(chunk, sample_rate):
    inputs = processor.feature_extractor(chunk, return_tensors="pt", sampling_rate=sample_rate)

    input_features = inputs["input_features"]
    attention_mask = torch.ones(input_features.shape, dtype=torch.long)  # All positions are valid since no padding is used

    # Generate transcription with increased max_new_tokens
    with torch.no_grad():
        # generated_ids = model.generate(
        #     input_features,
        #     attention_mask=attention_mask,
        #     max_new_tokens=444,          # Increased from default
        #     num_beams=5,                 # Using beam search for better quality
        #     temperature=0.0,             # Lower temperature for more focused sampling
        #     no_repeat_ngram_size=3,      # Avoid repeating same phrases
        #     length_penalty=1.0,          # Don't penalize longer outputs
        # )
        generated_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            max_new_tokens=444,          # Increased from default
            num_beams=5,                 # Using beam search for better quality
            temperature=0.0,             # Lower temperature for more focused sampling
            no_repeat_ngram_size=3,      # Avoid repeating same phrases
            length_penalty=1.0,          # Don't penalize longer outputs
            # repetition_penalty=2.0,
            # max_length=500  # Increase as needed
            
            # max_new_tokens=444,          # Increased from default
            # num_beams=5,                 # Using beam search for better quality
            # temperature=0.0,             # Lower temperature for more focused sampling
            # no_repeat_ngram_size=3,      # Avoid repeating same phrases
            # length_penalty=1.0,          # Don't penalize longer outputs
        )

    # Decode the transcription
    transcription = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True
    )[0]
    # print(transcription)
    return transcription


# # Generate transcription
# with torch.no_grad():
#     predicted_ids = model.generate(input_features, attention_mask=attention_mask)

# # Decode the transcription
# transcription = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
# print("Transcription:", transcription)

def merge_overlapping_transcriptions_improve(transcriptions, overlap_duration=2):
    """
    Merges overlapping transcriptions by handling overlaps carefully.

    Args:
    - transcriptions: List of transcription strings for overlapping chunks.
    - overlap_duration: Duration (in seconds) of the overlapping region.

    Returns:
    - A single, merged transcription string.
    """
    if not transcriptions:
        return ""

    merged_transcription = transcriptions[0]

    for i in range(1, len(transcriptions)):
        prev = merged_transcription.split()
        curr = transcriptions[i].split()

        # Find the overlap region by matching the end of `prev` with the start of `curr`
        overlap_index = 0
        for j in range(1, min(len(prev), len(curr)) + 1):
            if prev[-j:] == curr[:j]:
                overlap_index = j

        # Merge: Add only the non-overlapping part of `curr`
        merged_transcription += " " + " ".join(curr[overlap_index:])

    return merged_transcription


def merge_overlapping_transcriptions(transcriptions, overlap_duration=2, chunk_duration=30):
    """
    Merges overlapping transcriptions by removing duplicates.
    
    Args:
    - transcriptions: List of strings, transcriptions of overlapping chunks.
    - overlap_duration: Duration (in seconds) of the overlapping region.
    - chunk_duration: Duration (in seconds) of each chunk.
    
    Returns:
    - A single, merged transcription string.
    """
    merged_transcription = transcriptions[0]  # Start with the first chunk's transcription
    
    for i in range(1, len(transcriptions)):
        prev = transcriptions[i - 1]
        curr = transcriptions[i]
        
        # Find the overlapping region using string matching
        overlap_start = max(0, len(prev) - len(curr))
        while overlap_start > 0 and prev[overlap_start:] not in curr:
            overlap_start -= 1
        
        # Merge: Add only the non-overlapping part of the current chunk
        merged_transcription += curr[len(prev) - overlap_start:]
    
    return merged_transcription

def merge_transcriptions_by_words(transcriptions):
    """
    Merges transcriptions by identifying overlapping words.
    
    Args:
    - transcriptions: List of transcription strings.
    
    Returns:
    - Merged transcription string.
    """
    merged_transcription = transcriptions[0].split()
    
    for i in range(1, len(transcriptions)):
        prev_words = merged_transcription
        curr_words = transcriptions[i].split()
        
        # Find overlap by checking common suffix-prefix
        overlap_index = 0
        for j in range(1, min(len(prev_words), len(curr_words)) + 1):
            if prev_words[-j:] == curr_words[:j]:
                overlap_index = j
        
        # Merge by appending only the non-overlapping part
        merged_transcription.extend(curr_words[overlap_index:])
    
    return " ".join(merged_transcription)

from difflib import SequenceMatcher

def merge_with_similarity(transcriptions):
    """
    Merges transcriptions using sequence similarity.
    
    Args:
    - transcriptions: List of transcription strings.
    
    Returns:
    - Merged transcription string.
    """
    merged_transcription = transcriptions[0]
    
    for i in range(1, len(transcriptions)):
        prev = transcriptions[i - 1]
        curr = transcriptions[i]
        
        # Find the longest matching subsequence
        matcher = SequenceMatcher(None, prev, curr)
        match = matcher.find_longest_match(0, len(prev), 0, len(curr))
        
        # Merge: Add only the non-overlapping part of the current chunk
        if match.size > 0:
            merged_transcription += curr[match.b + match.size:]
        else:
            merged_transcription += " " + curr
    
    return merged_transcription


# Transcribe all chunks

transcriptions = []
for i, chunk in enumerate(audio_chunks):
    print(f"Transcribing chunk {i + 1}/{len(audio_chunks)}...")
    transcription = transcribe_chunk(chunk, rate)
    transcriptions.append(transcription)

# Combine all transcriptions into a single text
# final_transcription = " ".join(transcriptions)
# final_transcription = merge_overlapping_transcriptions(transcriptions, 2, 30)
final_transcription = merge_overlapping_transcriptions_improve(transcriptions, 2)
# final_transcription = merge_transcriptions_by_words(transcriptions)
# final_transcription = merge_with_similarity(transcriptions)
print("Final Transcription:", final_transcription)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

# sys.exit()


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

    # unsanitize = model.transcribe(source_audio)["text"]

    # Load the audio file
    audio_path = source_audio
    audio, rate = librosa.load(audio_path, sr=16000)  # Whisper expects 16 kHz audio

    # Split into 30-second chunks
    audio_chunks = split_audio_with_overlap(audio, rate, chunk_duration=30)

    # Transcribe all chunks

    transcriptions = []
    for i, chunk in enumerate(audio_chunks):
        print(f"Transcribing chunk {i + 1}/{len(audio_chunks)}...")
        transcription = transcribe_chunk(chunk, rate)
        transcriptions.append(transcription)

    # Combine all transcriptions into a single text
    # final_transcription = " ".join(transcriptions)
    # final_transcription = merge_overlapping_transcriptions(transcriptions, 2, 30)
    final_transcription = merge_overlapping_transcriptions_improve(transcriptions, 2)
    # final_transcription = merge_transcriptions_by_words(transcriptions)
    # final_transcription = merge_with_similarity(transcriptions)
    # print("Final Transcription:", final_transcription)

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
