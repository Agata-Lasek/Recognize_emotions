import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import tensorflow as tf
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

# Konfiguracja katalogów
tess_dir = r'C:/Users/smaga/Desktop/Inz/TESS/'
ravdess_dir = r'C:/Users/smaga/Desktop/Inz/Audio_Speech_Actors/'

emotion_directories = {
    "happy": ["OAF_happy", "YAF_happy"],
    "sad": ["OAF_sad", "YAF_sad"],
    "angry": ["OAF_angry", "YAF_angry"],
    "neutral": ["OAF_neutral", "YAF_neutral"],
    "disgust": ["OAF_disgust", "YAF_disgust"],
    "fear": ["OAF_fear", "YAF_fear"],
}

# Mapowanie emocji w RAVDESS z filtrowaniem i zamianą nazwy
ravdess_emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",       # Zamiana "fearful" na "fear"
    "07": "disgust"
}

import numpy as np

# Dodanie szumu
def add_noise(signal, noise_level=0.005):
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise

# Zmiana wysokości
def change_pitch(signal, sr, pitch_factor=2):
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=pitch_factor)

# Zmiana prędkości
def change_speed(signal, speed_factor=1.2):
    return librosa.effects.time_stretch(signal, rate=speed_factor)

# Funkcja augmentacji
def augment_signal(signal, sr):
    augmented_signals = []
    augmented_signals.append(signal)  # Oryginalny sygnał
    augmented_signals.append(add_noise(signal, noise_level=0.005))
    augmented_signals.append(change_pitch(signal, sr, pitch_factor=np.random.uniform(-3, 3)))
    augmented_signals.append(change_speed(signal, speed_factor=np.random.uniform(0.7, 1.3)))
    return augmented_signals


# Funkcja do odczytu plików RAVDESS
def load_ravdess_data(ravdess_dir, valid_emotions):
    ravdess_files = []
    ravdess_labels = []
    
    # Iteracja przez wszystkie foldery w `ravdess_dir`
    for actor_folder in os.listdir(ravdess_dir):
        actor_path = os.path.join(ravdess_dir, actor_folder)
        if os.path.isdir(actor_path):  # Sprawdzanie, czy to folder
            for file in os.listdir(actor_path):
                if file.endswith(".wav"):  # Tylko pliki `.wav`
                    # Wyciąganie emocji z nazwy pliku
                    emotion_id = file.split("-")[2]  # Trzeci segment w nazwie pliku
                    emotion = ravdess_emotion_map.get(emotion_id)
                    if emotion in valid_emotions:  # Upewnij się, że emocja jest w valid_emotions
                        file_path = os.path.join(actor_path, file)
                        ravdess_files.append(file_path)
                        ravdess_labels.append(emotion)
    
    return ravdess_files, ravdess_labels

# Ładowanie danych z RAVDESS (tylko wybrane emocje)
valid_emotions = {"neutral", "happy", "sad", "angry", "fear", "disgust"}
ravdess_files, ravdess_labels = load_ravdess_data(ravdess_dir, valid_emotions)

# Ładowanie danych z TESS
tess_files = []
tess_labels = []
for emotion, folders in emotion_directories.items():
    for folder in folders:
        folder_path = os.path.join(tess_dir, folder)
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".wav")]
        tess_files.extend(files)
        tess_labels.extend([emotion] * len(files))

# Łączenie danych z RAVDESS i TESS
all_files = tess_files + ravdess_files
all_labels = tess_labels + ravdess_labels

# Zakodowanie emocji
emotion_encoder = LabelEncoder()
encoded_labels = emotion_encoder.fit_transform(all_labels)
categorical_labels = to_categorical(encoded_labels)

# Normalizacja głośności
def normalize_volume(signal, target_db=-20):
    rms = np.sqrt(np.mean(signal**2))
    current_db = 20 * np.log10(rms)
    scalar = 10**((target_db - current_db) / 20)
    return signal * scalar
    

# Funkcja do wczytania plików i przetwarzania danych
def prepare_data(file_paths, emotions, max_pad_len=128, target_db=-20, use_mfcc=False, n_mfcc=13):
    X, y = [], []
    for file_path, emotion in zip(file_paths, emotions):
        # Wczytaj oryginalny sygnał
        signal, sr = librosa.load(file_path, sr=None)
        signal = normalize_volume(signal, target_db)
        
        # Augmentacja sygnału
        augmented_signals = augment_signal(signal, sr)
        
        for augmented_signal in augmented_signals:
            # Generowanie cech
            if use_mfcc:
                features = librosa.feature.mfcc(y=augmented_signal, sr=sr, n_mfcc=n_mfcc)
            else:
                spectrogram = librosa.feature.melspectrogram(y=augmented_signal, sr=sr)
                features = librosa.power_to_db(spectrogram, ref=np.max)
            
            # Normalizacja
            features = (features - np.mean(features)) / np.std(features)
            
            # Dopasowanie wymiarów
            if features.shape[1] > max_pad_len:
                features = features[:, :max_pad_len]
            else:
                pad_width = max_pad_len - features.shape[1]
                features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
            
            X.append(features)
            y.append(emotion)
    
    # Przekształć listy na numpy arrays
    X = np.array(X)
    X = np.expand_dims(X, axis=-1)  # Dodaj wymiar kanału
    y = np.array(y)
    return X, y

# Wizualizacja przykładowych cech: Mel-Spektrogram i MFCC
def plot_example(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    
    # Mel-spektrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # MFCC
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

    plt.figure(figsize=(12, 6))

    # Mel-spektrogram
    plt.subplot(1, 2, 1)
    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel-Spektrogram")

    # MFCC
    plt.subplot(1, 2, 2)
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title("MFCC")

    plt.tight_layout()
    plt.show()


# Przygotowanie danych
print("Przygotowanie danych dla Mel-Spektrogramów...")
X_mel, y_mel = prepare_data(all_files, categorical_labels, use_mfcc=False)

print("Przygotowanie danych dla MFCC...")
X_mfcc, y_mfcc = prepare_data(all_files, categorical_labels, use_mfcc=True)

# Podział danych na zestawy: Mel-Spektrogramy
X_train_mel, X_temp_mel, y_train_mel, y_temp_mel = train_test_split(X_mel, y_mel, test_size=0.3, random_state=42)
X_val_mel, X_test_mel, y_val_mel, y_test_mel = train_test_split(X_temp_mel, y_temp_mel, test_size=0.5, random_state=42)

# Podział danych na zestawy: MFCC
X_train_mfcc, X_temp_mfcc, y_train_mfcc, y_temp_mfcc = train_test_split(X_mfcc, y_mfcc, test_size=0.3, random_state=42)
X_val_mfcc, X_test_mfcc, y_val_mfcc, y_test_mfcc = train_test_split(X_temp_mfcc, y_temp_mfcc, test_size=0.5, random_state=42)

# Wyświetlenie rozmiarów zestawów danych
print("Rozmiary zestawów danych:")
print(f"Mel-Spektrogramy - Trening: {X_train_mel.shape}, Walidacja: {X_val_mel.shape}, Test: {X_test_mel.shape}")
print(f"MFCC - Trening: {X_train_mfcc.shape}, Walidacja: {X_val_mfcc.shape}, Test: {X_test_mfcc.shape}")

# Funkcja tworząca model
def create_model(input_shape, num_classes, trainable=True):
    model = Sequential([
        # Warstwa konwolucyjna 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Warstwa konwolucyjna 2
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        # Warstwa konwolucyjna 3
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01), trainable=trainable),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        # Spłaszczenie i warstwy gęste
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01), trainable=trainable),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', trainable=trainable)
    ])

    return model

# Etap 1: Wstępne trenowanie modelu z początkowym learning rate dla Mel-Spektrogramów
print("Rozpoczynam wstępne trenowanie na Mel-Spektrogramach...")
model_mel = create_model(X_train_mel.shape[1:], y_train_mel.shape[1])
model_mel.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history_mel = model_mel.fit(X_train_mel, y_train_mel, epochs=10, batch_size=32, validation_data=(X_val_mel, y_val_mel), verbose=1)

# Fine-tuning dla Mel-Spektrogramów
print("Rozpoczynam fine-tuning na Mel-Spektrogramach...")
for layer in model_mel.layers:
    layer.trainable = True
model_mel.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
fine_tune_history_mel = model_mel.fit(X_train_mel, y_train_mel, epochs=10, batch_size=32, validation_data=(X_val_mel, y_val_mel), verbose=1)

# Ocena modelu na Mel-Spektrogramach
test_loss_mel, test_accuracy_mel = model_mel.evaluate(X_test_mel, y_test_mel, verbose=1)
print(f"Mel-Spektrogramy - Test Loss: {test_loss_mel:.4f}")
print(f"Mel-Spektrogramy - Test Accuracy: {test_accuracy_mel:.4f}")

# Etap 1: Wstępne trenowanie modelu z początkowym learning rate dla MFCC
print("Rozpoczynam wstępne trenowanie na MFCC...")
model_mfcc = create_model(X_train_mfcc.shape[1:], y_train_mfcc.shape[1])
model_mfcc.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history_mfcc = model_mfcc.fit(X_train_mfcc, y_train_mfcc, epochs=10, batch_size=32, validation_data=(X_val_mfcc, y_val_mfcc), verbose=1)

# Fine-tuning dla MFCC
print("Rozpoczynam fine-tuning na MFCC...")
for layer in model_mfcc.layers:
    layer.trainable = True
model_mfcc.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
fine_tune_history_mfcc = model_mfcc.fit(X_train_mfcc, y_train_mfcc, epochs=10, batch_size=32, validation_data=(X_val_mfcc, y_val_mfcc), verbose=1)

# Ocena modelu na MFCC
test_loss_mfcc, test_accuracy_mfcc = model_mfcc.evaluate(X_test_mfcc, y_test_mfcc, verbose=1)
print(f"MFCC - Test Loss: {test_loss_mfcc:.4f}")
print(f"MFCC - Test Accuracy: {test_accuracy_mfcc:.4f}")

# Porównanie wyników
def plot_fine_tune_comparison(history1, history2, fine_tune_history1, fine_tune_history2, labels=("Mel-Spektrogramy", "MFCC")):
    plt.figure(figsize=(12, 6))
    
    # Dokładność
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['accuracy'] + fine_tune_history1.history['accuracy'], label=f'{labels[0]} - Train Accuracy')
    plt.plot(history1.history['val_accuracy'] + fine_tune_history1.history['val_accuracy'], label=f'{labels[0]} - Val Accuracy')
    plt.plot(history2.history['accuracy'] + fine_tune_history2.history['accuracy'], label=f'{labels[1]} - Train Accuracy')
    plt.plot(history2.history['val_accuracy'] + fine_tune_history2.history['val_accuracy'], label=f'{labels[1]} - Val Accuracy')
    plt.title("Porównanie dokładności wstępnego trenowania i fine-tuningu")
    plt.xlabel("Epoka")
    plt.ylabel("Dokładność")
    plt.legend()
    plt.grid()

    # Strata
    plt.subplot(1, 2, 2)
    plt.plot(history1.history['loss'] + fine_tune_history1.history['loss'], label=f'{labels[0]} - Train Loss')
    plt.plot(history1.history['val_loss'] + fine_tune_history1.history['val_loss'], label=f'{labels[0]} - Val Loss')
    plt.plot(history2.history['loss'] + fine_tune_history2.history['loss'], label=f'{labels[1]} - Train Loss')
    plt.plot(history2.history['val_loss'] + fine_tune_history2.history['val_loss'], label=f'{labels[1]} - Val Loss')
    plt.title("Porównanie straty wstępnego trenowania i fine-tuningu")
    plt.xlabel("Epoka")
    plt.ylabel("Strata")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Wizualizacja porównania
plot_fine_tune_comparison(history_mel, history_mfcc, fine_tune_history_mel, fine_tune_history_mfcc)
