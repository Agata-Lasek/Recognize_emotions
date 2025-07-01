# Rozpoznawanie Emocji w Mowie

## Opis projektu
Celem projektu było zaprojektowanie i zaimplementowanie aplikacji webowej, która rozpoznaje emocje użytkownika na podstawie sygnału mowy. 
Aplikacja integruje rozpoznawanie mowy oraz analizę tekstu, co pozwala na bardziej precyzyjne określenie emocji i generowanie odpowiedzi dopasowanych do stanu emocjonalnego rozmówcy.

## Funkcjonalności

- Zamiana mowy na tekst (ASR)
- Analiza emocji w sygnale dźwiękowym
- Analiza emocjonalna tekstu
- Łączenie wyników z obu analiz
- Generowanie odpowiedzi emocjonalnych
- Interfejs webowy zbudowany w HTML/CSS/JS
- Backend w Pythonie z użyciem Flask


## Dane wejściowe i ich przygotowanie

### Zestaw nEMO
- Źródło: dane z [Hugging Face – nEMO Dataset](https://huggingface.co/datasets/amu-cai/nEMO)
- Autor: Christop, Iwona
- Licencja: Creative Commons CC BY-NC-SA 4.0  
- Zawartość: ~3h nagrań (9 lektorów, 6 emocji, ~700 próbki/emocję)  
- Podział: 70% trening, 15% walidacja, 15% test

### Augmentacja danych
Aby zwiększyć różnorodność i odporność modelu:
- **Dodawanie białego szumu**  
- **Zmiana tonacji** (±2 półtony)  
- **Losowe przesunięcia czasowe**

## Sieć neuronowa do klasyfikacji emocji

### Wstępne przetwarzanie
1. **Normalizacja głośności** – wyrównanie RMS wszystkich nagrań.  
2. **Przycinanie i padding** do stałej długości 3 sek.  

### Architektura CNN
- **Blok konwolucyjny ×3**:  
  - filtr 3×3, padding = ‘same’,  
  - LeakyReLU, BatchNorm, MaxPooling2D  
  - Dropout (0.3–0.4)  
- **Flatten → Dense(256, LeakyReLU) → Dropout(0.4) → Dense(6, Softmax)**  

### Trening i walidacja
- Optymalizator: **Adam** (lr=1e‑4)  
- Funkcja straty: **Categorical Crossentropy**  
- Metryki: accuracy, f1‑score  
- Early stopping na podstawie walidacji (patience=10)

### Model Wav2Vec2

- Źródło: ([https://doi.org/10.48550/arXiv.2006.11477](https://doi.org/10.48550/arXiv.2006.11477))
- Proces:
  1. Ładowanie audio (Librosa, konwersja MP3→WAV)  
  2. Tokenizacja (Wav2Vec2Processor)  
  3. Dekodowanie algorytmem CTC  

### Własny prototyp STT
- Próba własnej sieci CNN do ASR (spectrogram + MFCC)  
- Wynik: ~60% dokładności → nie włączono do finalnej aplikacji  

## Analiza tekstu i klasyfikacja semantyczna

### Baza NAWL
- **Normative Affect and Word List (NAWL)** – słowa z przypisanymi wartościami emocjonalnymi  
- Link: (https://lobi.nencki.edu.pl/research/18/)
### Rozszerzenie form fleksyjnych z PELCRA
- Analiza korpusu NKJP przez [PELCRA](http://pelcra.pl/)  
- Dodanie najczęstszych form odmienionych do bazy  

### Łączenie wyników akustycznych i tekstowych
- Wynik akustyczny: waga 90%  
- Wynik semantyczny: waga 10%  
- Finalna decyzja: klasyfikacja emocji i generacja odpowiedzi

## Interfejs użytkownika
- **Frontend**: HTML5, CSS3, JavaScript  
- Strona pozwala na:  
  - Nagranie lub załadowanie pliku audio  
  - Wyświetlenie transkrypcji i wykresu emocji  
  - Otrzymanie dynamicznej odpowiedzi  





Citation Information:
- **nEMO**
@inproceedings{christop-2024-nemo-dataset,
    title = "n{EMO}: Dataset of Emotional Speech in {P}olish",
    author = "Christop, Iwona",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1059",
    pages = "12111--12116",
    abstract = "Speech emotion recognition has become increasingly important in recent years due to its potential applications in healthcare, customer service, and personalization of dialogue systems. However, a major issue in this field is the lack of datasets that adequately represent basic emotional states across various language families. As datasets covering Slavic languages are rare, there is a need to address this research gap. This paper presents the development of nEMO, a novel corpus of emotional speech in Polish. The dataset comprises over 3 hours of samples recorded with the participation of nine actors portraying six emotional states: anger, fear, happiness, sadness, surprise, and a neutral state. The text material used was carefully selected to represent the phonetics of the Polish language adequately. The corpus is freely available under the terms of a Creative Commons license (CC BY-NC-SA 4.0).",
}

- **Wav2Vec2**
@misc{wav2vec,
author ={Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli},
  title={wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations},
  howpublished={\url{https://doi.org/10.48550/arXiv.2006.11477}},
  month = {Data dostępu: listopad},
  year = {2024}
}

- **NAWL**
@article{nawl1, 
    author = {Project Leader: LOBI, Laboratory of Brain Imaging}, 
    title = {Nencki Affective Word List (NAWL)}, 
    month = {Data dostępu: listopad},
    year = {2024},
    howpublished = {\url{https://lobi.nencki.edu.pl/research/18/}}

}

- **PELCRA**
@misc{PELCRA,
author ={Piotr Pęzik},
title = 	 { (2012) Wyszukiwarka PELCRA dla danych NKJP. Narodowy Korpus Języka Polskiego. Przepiórkowski A., Bańko M., Górski R., Lewandowska-Tomaszczyk B. (red.). 2012. Wydawnictwo PWN .},
month = {Data dostępu: listopad},
year = {2024},
howpublished = {\url{https://nkjp.pl/}}
  }

