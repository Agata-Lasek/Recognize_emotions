import pandas as pd
import os
import random
import platform
import subprocess

import librosa
import numpy as np
from tqdm import tqdm

from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, TimeDistributed, Masking

tf.config.run_functions_eagerly(True)


# Ścieżka główna do danych i folderu `clips`
base_path = r'C:/Users/smaga/Desktop/Inz/pl'
clips_folder = os.path.join(base_path, 'clips')

# Wczytanie plików `train`, `dev` i `test`
train = pd.read_csv(os.path.join(base_path, 'train.tsv'), sep='\t')
dev = pd.read_csv(os.path.join(base_path, 'dev.tsv'), sep='\t')
test = pd.read_csv(os.path.join(base_path, 'test.tsv'), sep='\t')

# Wybór tylko najistotniejszych kolumn: `path` (do audio) i `sentence` (transkrypcja)
train = train[['path', 'sentence']]
dev = dev[['path', 'sentence']]
test = test[['path', 'sentence']]

# Dodanie pełnych ścieżek do plików audio
train['audio_path'] = train['path'].apply(lambda x: os.path.join(clips_folder, x))
dev['audio_path'] = dev['path'].apply(lambda x: os.path.join(clips_folder, x))
test['audio_path'] = test['path'].apply(lambda x: os.path.join(clips_folder, x))

# Opcjonalne: wczytanie pliku z długością nagrań i filtrowanie na podstawie długości
durations = pd.read_csv(os.path.join(base_path, 'clip_durations.tsv'), sep='\t')
train = train.merge(durations, left_on='path', right_on='clip', how='left')
train = train[train['duration[ms]'] < 10000]  # Filtrowanie na przykładzie nagrań krótszych niż 10 sekund

# Wypisanie przykładowych danych, by zweryfikować wczytanie
print(train.head())

# Wybór losowego nagrania
random_row = train.sample(1).iloc[0]

# Wyświetl ścieżkę do audio i transkrypcję
print("Ścieżka audio:", random_row['audio_path'])
print("Transkrypcja:", random_row['sentence'])

# Otwórz plik audio w domyślnym odtwarzaczu
audio_path = random_row['audio_path']
if platform.system() == "Windows":
    os.startfile(audio_path)
else:  # Linux
    subprocess.call(["xdg-open", audio_path])


# Funkcja do ekstrakcji MFCC
def extract_mfcc(audio_path, n_mfcc=13, sr=16000, duration=None):
    """
    Ekstrakcja MFCC z pliku audio.
    
    Parameters:
    - audio_path: ścieżka do pliku audio
    - n_mfcc: liczba współczynników MFCC
    - sr: docelowa częstotliwość próbkowania
    - duration: czas trwania w sekundach (jeśli ograniczamy długość sygnału)
    
    Returns:
    - Numpy array z MFCC
    """
    try:
        # Wczytaj audio z określoną częstotliwością próbkowania
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        # Ekstrakcja MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc
    except Exception as e:
        print(f"Błąd przy przetwarzaniu {audio_path}: {e}")
        return None

# Przetwarzanie zbioru treningowego
train['mfcc'] = None

# Iteruj przez zbiór danych i generuj MFCC
for i, row in tqdm(train.iterrows(), total=len(train), desc="Ekstrakcja MFCC dla treningu"):
    mfcc = extract_mfcc(row['audio_path'])
    if mfcc is not None:
        train.at[i, 'mfcc'] = mfcc.tolist()  # Konwersja do listy, aby zapisać w DataFrame

# Zapis do pliku pickle dla szybkiego wczytywania w przyszłości
train.to_pickle(os.path.join(base_path, 'train_with_mfcc.pkl'))

# Wczytanie zapisanego pliku
train_with_mfcc = pd.read_pickle(os.path.join(base_path, 'train_with_mfcc.pkl'))

# Wypisanie MFCC dla losowego przykładu
sample_mfcc = train_with_mfcc.sample(1)['mfcc'].values[0]
print("Przykładowe MFCC:")
print(np.array(sample_mfcc))

import librosa.display
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
librosa.display.specshow(np.array(sample_mfcc), x_axis='time', sr=16000)
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC')
plt.tight_layout()
plt.show()



# Maksymalna długość MFCC (np. dla najdłuższego nagrania w zbiorze)
MAX_MFCC_LENGTH = 100  # Dostosowane do najdłuższego nagrania w danych

# Funkcja do pad/truncate MFCC
def pad_mfcc(mfcc, max_length=MAX_MFCC_LENGTH):
    if mfcc.shape[1] > max_length:
        return mfcc[:, :max_length]
    else:
        return np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')

# Przygotowanie danych wejściowych (MFCC)
X_train = np.array([pad_mfcc(np.array(mfcc)) for mfcc in train_with_mfcc['mfcc']])

# Kodowanie transkrypcji na indeksy
le = LabelEncoder()
y_train = [list(sentence) for sentence in train_with_mfcc['sentence']]
y_train_encoded = [le.fit_transform(chars) for chars in y_train]

# Maksymalna długość transkrypcji
MAX_TEXT_LENGTH = 50  # Możesz dostosować na podstawie danych

# Pad transkrypcji (dla RNN/Transformer)
y_train_padded = pad_sequences(y_train_encoded, maxlen=MAX_TEXT_LENGTH, padding='post')

# Zamiana na NumPy array
X_train = np.array(X_train)
y_train_padded = np.array(y_train_padded)

# Dodanie wymiaru na czas dla danych wyjściowych
y_train_onehot = y_train_padded[..., np.newaxis]

# Parametry modelu
INPUT_DIM = X_train.shape[1:]  # Kształt danych wejściowych (czas, cechy MFCC)
OUTPUT_DIM = len(le.classes_)  # Liczba unikalnych znaków w transkrypcji

num_classes = len(le.classes_)
print(f"Liczba unikalnych klas: {num_classes}")
print(f"Etykiety: {le.classes_}")


# Budowa modelu
model = Sequential([
    Input(shape=INPUT_DIM),
    Masking(mask_value=0.0),
    LSTM(128, return_sequences=True, activation='relu'),
    TimeDistributed(Dense(OUTPUT_DIM, activation='softmax'))
])


# Kompilacja modelu
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Wyświetlenie modelu
model.summary()

# Dopasowanie y_train_onehot do liczby kroków czasowych wyjścia
TIME_STEPS = X_train.shape[1]  # Liczba kroków czasowych w danych wejściowych (MFCC po paddingu)

# Rozciągnięcie/przycięcie y_train_onehot do odpowiedniej liczby kroków czasowych
y_train_onehot_resized = pad_sequences(
    y_train_onehot.squeeze(),  # Usuń ostatni wymiar (1) przed przetwarzaniem
    maxlen=TIME_STEPS,  # Dopasowanie do liczby kroków czasowych
    padding='post'
)[..., np.newaxis]  # Przywróć wymiar 1, aby dane były w kształcie (batch_size, time_steps, 1)

# Zamiana na NumPy array (jeśli jeszcze nie jest)
y_train_onehot_resized = np.array(y_train_onehot_resized)

# Teraz `y_train_onehot_resized` ma dopasowany kształt
print("Kształt dopasowanego y_train_onehot:", y_train_onehot_resized.shape)

# Zastąpienie oryginalnego `y_train_onehot` zmienioną wersją
y_train_onehot = y_train_onehot_resized

for y in y_train_encoded:
    if max(y) >= num_classes:
        print(f"Błąd: Znaleziono wartość {max(y)} przekraczającą zakres {num_classes - 1}")


# Funkcja trenowania modelu z dekoratorem @tf.function
@tf.function
def train_model():
    history = model.fit(
        X_train, 
        y_train_onehot, 
        batch_size=32, 
        epochs=20, 
        validation_split=0.2
    )
    return history

# Uruchomienie treningu
history = train_model()


# Funkcja dekodująca predykcje
def decode_prediction(predictions):
    return ''.join(le.inverse_transform(np.argmax(predictions, axis=-1)))

# Przetestowanie pojedynczego przykładu
sample_index = 0
sample_mfcc = X_train[sample_index]
predicted = model.predict(sample_mfcc[np.newaxis, ...])
decoded_text = decode_prediction(predicted[0])

print("Oczekiwane:", ''.join(train_with_mfcc['sentence'].iloc[sample_index]))
print("Przewidywane:", decoded_text)
