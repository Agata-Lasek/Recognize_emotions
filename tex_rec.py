import os
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa

# Ścieżki do danych
test_dir = r"C:/Users/smaga/Desktop/Inz/test_samples"
emotions_csv = r"C:/Users/smaga/Desktop/Inz/nawl-analysis_with_percentages.csv"

# Wczytaj dane emocjonalne i przygotuj słownik dla szybkiego dostępu
emotions_df = pd.read_csv(emotions_csv, encoding="latin1")
emotions_dict = {row["word"]: {
    "happiness": row["happiness (%)"],
    "anger": row["anger (%)"],
    "sadness": row["sadness (%)"],
    "fear": row["fear (%)"],
} for _, row in emotions_df.iterrows()}

# Ładowanie modelu Wav2Vec2
print("Ładowanie modelu Wav2Vec2 dla języka polskiego...")
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-polish")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-polish").to("cuda" if torch.cuda.is_available() else "cpu")

# Funkcja do transkrypcji pliku audio
def transcribe_audio(file_path):
    try:
        # Wczytanie audio
        speech, rate = librosa.load(file_path, sr=16000)
        input_values = processor(speech, sampling_rate=rate, return_tensors="pt").input_values.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Transkrypcja
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        return transcription.lower()
    except Exception as e:
        print(f"Błąd podczas transkrypcji: {e}")
        return None

# Funkcja przypisania emocji i obliczania średnich
def analyze_emotions(text):
    words = text.split()
    emotion_totals = {emotion: 0.0 for emotion in ["happiness", "anger", "sadness", "fear", "neutral"]}
    word_count = {emotion: 0 for emotion in emotion_totals}

    for word in words:
        if word in emotions_dict:
            # Pobierz wartości emocji dla słowa
            values = emotions_dict[word]
            # Sprawdź, czy któraś emocja przekracza 30%
            max_emotion = max(values.values())
            if max_emotion > 30:
                for emotion, value in values.items():
                    emotion_totals[emotion] += value
                    word_count[emotion] += 1
            else:
                # Słowo uznane za neutralne
                emotion_totals["neutral"] += 100
                word_count["neutral"] += 1
        else:
            # Słowo nieznane - rozdziel emocje po równo
            for emotion in ["happiness", "anger", "sadness", "fear","neutral"]:
                emotion_totals[emotion] += 20
                word_count[emotion] += 1

    # Oblicz średnie
    averages = {emotion: (emotion_totals[emotion] / word_count[emotion] if word_count[emotion] > 0 else 0.0)
                for emotion in emotion_totals}

    return averages

# Przetwarzanie plików audio
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.wav', '.mp3'))]

for file in test_files:
    print(f"Rozpoznawanie tekstu dla: {file}")
    text = transcribe_audio(file)
    if text:
        print(f"Rozpoznany tekst: {text}")
        averages = analyze_emotions(text)
        print("Średnie wartości emocji:")
        for emotion, value in averages.items():
            print(f"  {emotion}: {value:.2f}%")
    else:
        print(f"Nie udało się rozpoznać tekstu dla pliku {file}")
