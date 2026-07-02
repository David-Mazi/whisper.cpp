import whisper
import torch
import os
import librosa
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

# Load the Whisper model
model = whisper.load_model("tiny.en")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Dataset
class CustomWhisperDataset(Dataset):
    def __init__(self, data_dir, sampling_rate=16000):
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.audio_files = []
        self.transcriptions = []
        
        # Collect all the audio files and their transcriptions
        audio_files = os.listdir(os.path.join(data_dir, 'audios'))
        for file_name in audio_files:
            if file_name.endswith('.mp3'):
                audio_path = os.path.join(data_dir, 'audio', file_name)
                transcription_path = os.path.join(data_dir, 'transcriptions', file_name.replace('.mp3', '.txt'))
                
                with open(transcription_path, 'r') as f:
                    transcription = f.read().strip()
                
                self.audio_files.append(audio_path)
                self.transcriptions.append(transcription)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        transcription = self.transcriptions[idx]
        
        # Load the audio
        audio, _ = librosa.load(audio_path, sr=self.sampling_rate)
        
        # Get audio features (log-Mel spectrogram) using Whisper's feature extraction
        audio_features = whisper.pad_or_trim(model.encode_audio(audio))

        # Prepare transcription as target
        # Whisper uses a byte pair encoding, so you'll have to encode the transcription to ids
        # This part will depend on the specific tokenizer or method you're using for your task
        # Assuming simple tokenization or processing as required

        # Assuming this placeholder step for tokenizing the transcription
        # `transcription_ids` should be converted into token IDs like the processor would
        # Handle it for actual Hugging Face models
        transcription_ids = whisper.tokenize(transcription)  # Adjust as per your tokenizer

        return audio_features, transcription_ids

# Load dataset
data_dir = './'
train_dataset = CustomWhisperDataset(data_dir)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (input_values, target) in enumerate(train_dataloader):
        input_values = input_values.to(device)
        target = target.to(device)
        
        # Forward pass
        loss = model(input_values=input_values, labels=target).loss
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    print(f"Epoch {epoch + 1} finished. Total Loss: {total_loss:.4f}")

# Save the fine-tuned model
model.save_pretrained('./whisper_finetuned')
