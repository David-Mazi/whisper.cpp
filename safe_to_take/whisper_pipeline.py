from transformers import pipeline

whisper_asr = pipeline(
    "automatic-speech-recognition", 
    model="./whisper-tiny.en-vox",
    # model="openai/whisper-tiny",
    chunk_length_s=30,
    generate_kwargs={
        # "max_length": 448,        # Allow for longer outputs
        # "temperature": 0.7,       # Add some randomness
        # "length_penalty": 1.0,    # Neutral penalty for output length
        # "repetition_penalty": 1.2 # Penalize repeated tokens
        "max_new_tokens": 444,          # Increased from default
        "num_beams": 5,                 # Using beam search for better quality
        "temperature": 0.0,             # Lower temperature for more focused sampling
        "no_repeat_ngram_size": 3,      # Avoid repeating same phrases
        "length_penalty": 1.0
    }
)
print(whisper_asr("glen.wav")["text"])