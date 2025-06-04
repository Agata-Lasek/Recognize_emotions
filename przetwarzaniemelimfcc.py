import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Ścieżki do folderów
base_path = r'C:/Users/smaga/Desktop/Inz/pl'
clips_folder = os.path.join(base_path, 'clips')
mel_original_folder = os.path.join(base_path, 'mel_original')
mfcc_original_folder = os.path.join(base_path, 'mfcc_original')

# Wczytanie danych z merged_data_2.tsv
merged_data_path = os.path.join(base_path, 'merged_data_2.tsv')
merged_data = pd.read_csv(merged_data_path, sep='\t')

# Ensure 'spectrogram' and 'mfcc' columns are added before 'padded_sentence' if they do not exist
if 'padded_sentence' in merged_data.columns:
    insert_pos = merged_data.columns.get_loc('padded_sentence')
else:
    insert_pos = len(merged_data.columns)  # Default to the end if 'padded_sentence' is not found

if 'spectrogram' not in merged_data.columns:
    merged_data.insert(insert_pos, 'spectrogram', None)

if 'mfcc' not in merged_data.columns:
    merged_data.insert(insert_pos, 'mfcc', None)

# Funkcja do obliczania spektrogramu Mel
def compute_mel_spectrogram(audio_path, sr=22050, n_mels=128, fmax=8000):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    except Exception as e:
        print(f"Błąd podczas przetwarzania pliku {audio_path}: {e}")
        return None

# Funkcja do obliczania MFCC
def compute_mfcc(audio_path, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return mfcc
    except Exception as e:
        print(f"Błąd podczas przetwarzania pliku {audio_path}: {e}")
        return None

# Przetwarzanie nagrań
processed_mel_count = 0
processed_mfcc_count = 0

for index, row in merged_data.iterrows():
    audio_file = os.path.join(clips_folder, row['path'])
    if not os.path.exists(audio_file):
        print(f"Plik audio nie istnieje: {audio_file}")
        continue

    mel_output_path = os.path.join(mel_original_folder, f"{row['path']}.npy")
    mfcc_output_path = os.path.join(mfcc_original_folder, f"{row['path']}_mfcc.npy")

    mel_exists = os.path.exists(mel_output_path)
    mfcc_exists = os.path.exists(mfcc_output_path)

    # Update paths if files already exist
    if mel_exists:
        merged_data.at[index, 'spectrogram'] = mel_output_path
    if mfcc_exists:
        merged_data.at[index, 'mfcc'] = mfcc_output_path

    # Skip processing if both files exist
    if mel_exists and mfcc_exists:
        continue

    # Compute and save Mel spectrogram if it doesn't exist
    if not mel_exists:
        mel_spec = compute_mel_spectrogram(audio_file)
        if mel_spec is not None:
            np.save(mel_output_path, mel_spec)
            merged_data.at[index, 'spectrogram'] = mel_output_path
            processed_mel_count += 1

    # Compute and save MFCC if it doesn't exist
    if not mfcc_exists:
        mfcc = compute_mfcc(audio_file)
        if mfcc is not None:
            np.save(mfcc_output_path, mfcc)
            merged_data.at[index, 'mfcc'] = mfcc_output_path
            processed_mfcc_count += 1

# Zapisanie zaktualizowanego pliku merged_data_2.tsv
merged_data.to_csv(merged_data_path, sep='\t', index=False, encoding='utf-8')

print(f"Zaktualizowano plik: {merged_data_path}")
print(f"Liczba nowych spektrogramów Mel: {processed_mel_count}")
print(f"Liczba nowych plików MFCC: {processed_mfcc_count}")
