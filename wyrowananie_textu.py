import os
import pandas as pd
import csv

# Ścieżka do danych i folderu clips
base_path = r'C:/Users/smaga/Desktop/Inz/pl'
clips_folder = os.path.join(base_path, 'clips')

# Wczytanie pliku train.tsv z uwzględnieniem cytowania
train = pd.read_csv(
    os.path.join(base_path, 'validated.tsv'),
    sep='\t',
    quoting=csv.QUOTE_NONE,
    engine='python',
    on_bad_lines='skip',
)

# Wczytanie pliku clip_durations.tsv
durations = pd.read_csv(os.path.join(base_path, 'clip_durations.tsv'), sep='\t')

# Zmiana nazw kolumn dla ujednolicenia (jeśli nazwy różnią się między plikami)
durations.rename(columns={'clip': 'path', 'duration[ms]': 'duration_ms'}, inplace=True)

# Łączenie danych na podstawie nazwy nagrania
merged_data = pd.merge(
    train[['path', 'sentence']],
    durations,
    on='path',
    how='inner'
)

# Dodanie kolumny z liczbą znaków w zdaniu
merged_data['chars'] = merged_data['sentence'].str.len()

# Przekonwertowanie czasu trwania z milisekund na sekundy
merged_data['duration'] = merged_data['duration_ms'] 

# Znalezienie maksymalnej liczby znaków w zdaniu
max_length = merged_data['chars'].max()
print(f"Maksymalna liczba znaków w zdaniu: {max_length}")

# Dodanie nowej kolumny z przedłużonymi zdaniami
merged_data['padded_sentence'] = merged_data['sentence'].apply(
    lambda x: x + "_" * (max_length - len(x)) if len(x) < max_length else x
)

# Wybranie interesujących nas kolumn
final_data = merged_data[['path', 'duration', 'sentence', 'chars', 'padded_sentence']]

# Zapisanie do nowego pliku TSV
output_path = os.path.join(base_path, 'merged_data_2.tsv')
final_data.to_csv(output_path, sep='\t', index=False, encoding='utf-8')

print(f"Plik został zapisany: {output_path}")

# Wyświetlenie przykładowych danych
examples = final_data.head(6)
print("\nPierwsze 6 przykładów:")
print(examples)
