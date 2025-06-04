import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Ścieżki do folderów
original_mel_folder = r'C:/Users/smaga/Desktop/Inz/pl/mel_original'
padded_mel_folder = r'C:/Users/smaga/Desktop/Inz/pl/mel_spectrograms'
updated_mel_folder = r'C:/Users/smaga/Desktop/Inz/pl/mel_updated'
original_mfcc_folder = r'C:/Users/smaga/Desktop/Inz/pl/mfcc_original'
padded_mfcc_folder = r'C:/Users/smaga/Desktop/Inz/pl/mfcc'

# Wczytanie danych z przykładowego pliku
example_file = os.listdir(updated_mel_folder)[0]  # Wybór przykładowego pliku
original_mel_file = os.path.join(original_mel_folder, example_file)
padded_mel_file = os.path.join(padded_mel_folder, example_file)
updated_mel_file = os.path.join(updated_mel_folder, example_file)

# Ścieżki dla MFCC
original_mfcc_file = os.path.join(original_mfcc_folder, example_file.replace('.npy', '_mfcc.npy'))
padded_mfcc_file = os.path.join(padded_mfcc_folder, example_file.replace('.npy', '_mfcc.npy'))

# Wczytanie danych Mel
original_mel = np.load(original_mel_file).astype(np.float32)
padded_mel = np.load(padded_mel_file).astype(np.float32)
updated_mel = np.load(updated_mel_file).astype(np.float32)

# Wczytanie danych MFCC
original_mfcc = np.load(original_mfcc_file).astype(np.float32)
padded_mfcc = np.load(padded_mfcc_file).astype(np.float32)

# Wczytanie zdania i jego rozszerzonej wersji
sentence = "Przykładowe zdanie"  # Tutaj dodaj właściwe zdanie związane z plikiem
extended_sentence = sentence + "_" * (len(padded_mel[0]) - len(sentence))

# Wyświetlanie informacji o pliku
print(f"Nazwa nagrania: {example_file}")
print(f"Oryginalne zdanie: {sentence} (długość: {len(sentence)})")
print(f"Rozszerzone zdanie: {extended_sentence} (długość: {len(extended_sentence)})")

# Wyświetlenie porównania Mel
plt.figure(figsize=(15, 5))

# Oryginalny Mel
plt.subplot(1, 2, 1)
librosa.display.specshow(original_mel, sr=22050, x_axis='time', y_axis='mel', fmax=8000, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title("Spektrogram Mel (oryginalny)")
plt.xlabel("Czas")
plt.ylabel("Częstotliwość (Mel)")

# Zaktualizowany Mel
plt.subplot(1, 2, 2)
librosa.display.specshow(updated_mel, sr=22050, x_axis='time', y_axis='mel', fmax=8000, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title("Spektrogram Mel (zaktualizowany)")
plt.xlabel("Czas")
plt.ylabel("Częstotliwość (Mel)")

plt.tight_layout()
plt.show()

# Statystyki danych Mel
print("Statystyki danych dla oryginalnego Mel:")
print(f"Min: {original_mel.min()}, Max: {original_mel.max()}, Średnia: {original_mel.mean()}")

print("\nStatystyki danych dla zaktualizowanego Mel:")
print(f"Min: {updated_mel.min()}, Max: {updated_mel.max()}, Średnia: {updated_mel.mean()}")

# Wyświetlenie porównania MFCC
plt.figure(figsize=(15, 5))

# Oryginalne MFCC
plt.subplot(1, 2, 1)
librosa.display.specshow(original_mfcc, sr=22050, x_axis='time', cmap='viridis')
plt.colorbar()
plt.title("MFCC (oryginalne)")
plt.xlabel("Czas")
plt.ylabel("MFCC")

# Padding MFCC
plt.subplot(1, 2, 2)
librosa.display.specshow(padded_mfcc, sr=22050, x_axis='time', cmap='viridis')
plt.colorbar()
plt.title("MFCC (wydłużone)")
plt.xlabel("Czas")
plt.ylabel("MFCC")

plt.tight_layout()
plt.show()

# Statystyki danych MFCC
print("\nStatystyki danych dla oryginalnego MFCC:")
print(f"Min: {original_mfcc.min()}, Max: {original_mfcc.max()}, Średnia: {original_mfcc.mean()}")

print("\nStatystyki danych dla wydłużonego MFCC:")
print(f"Min: {padded_mfcc.min()}, Max: {padded_mfcc.max()}, Średnia: {padded_mfcc.mean()}")
