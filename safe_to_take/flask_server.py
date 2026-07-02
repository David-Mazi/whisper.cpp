from flask import Flask, request, jsonify
import whisper
import os

app = Flask(__name__)
LoadedModel = whisper.load_model("tiny.en")
from transformers import pipeline

whisper_asr = pipeline(
    "automatic-speech-recognition", 
    model="./whisper-tiny.en-vox",
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

import subprocess

def convert_aac_to_whisper_wav(input_path, output_path):
    """
    Convert an AAC file to a WAV format optimized for OpenAI Whisper.

    Args:
        input_path (str): The path to the input .aac file.
        output_path (str): The path to save the output .wav file.

    Returns:
        bool: True if the conversion was successful, False otherwise.
    """
    try:
        # Build the FFmpeg command with optimal settings
        command = [
            "ffmpeg",
            "-i", input_path,        # Input file
            "-ac", "1",              # Downmix to mono
            "-ar", "16000",          # Set sample rate to 16 kHz
            "-acodec", "pcm_s16le",  # Use 16-bit PCM format
            output_path              # Output file
        ]

        # Run the FFmpeg command
        subprocess.run(command, check=True)
        
        print(f"Conversion successful! WAV file saved at: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        return False

# # Example usage:
# input_aac = "example.aac"  # Path to your .aac file
# output_wav = "example_whisper.wav" # Path to save the .wav file
# convert_aac_to_whisper_wav(input_aac, output_wav)


# Directory to save uploaded files
DOWNLOAD_FOLDER = './downloads'
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

@app.route('/omniaai/loadmodel', methods=['GET'])
def load_model():
    # Access query parameters
    model = request.args.get('model', 'base.en')

    global LoadedModel
    LoadedModel = whisper.load_model(model)

    return f'Model Loaded: {model}'

@app.route('/api/data', methods=['GET'])
def get_data():
    # Access query parameters
    name = request.args.get('name')
    
    # Process the request and create a response
    data = {'message': f'Hello, {name}!'}
    return jsonify(data)

@app.route('/ai/whisper', methods=['GET'])
def transcribe():
    # Access query parameters
    path = request.args.get('path', 'test_audio.wav')
    global LoadedModel
    if isinstance(LoadedModel, str):
        return "No Model Loaded, use loadmodel endpoint to load one"
    
    result = LoadedModel.transcribe(path)

    unsanitize = result["text"]
    # unsanitize = unsanitize.replace(" slash ", "/")
    # unsanitize = unsanitize.replace("-slash-", "/")
    # unsanitize = unsanitize.replace(" over ", "/")
    # sanitized = unsanitize.replace("-over-", "/")
    return unsanitize

@app.route('/ai/upload', methods=['POST'])
def save_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        # Save the file
        file_path = os.path.join(DOWNLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Process the file (for example, read its metadata or re-encode it)
        # Placeholder: Just print the file path
        print(f"File saved to: {file_path}")

        # result = LoadedModel.transcribe(file_path)
        result = whisper_asr(file_path)
        
        transcription = result["text"]

        return jsonify({"message": f"File uploaded successfully", "transcription": transcription}), 200

    return jsonify({"error": "Unknown error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)
