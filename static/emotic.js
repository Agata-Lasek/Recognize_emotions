const emotionUnicode = {
    happy: "😀",
    angry: "😡",
    sad: "😢",
    surprise: "😲",
    neutral: "😐",
    fear: "😨",
    disgust: "🤢"
};

// Funkcja do animacji reakcji na emocję
function showEmotionReaction(emotion) {
    const overlay = document.getElementById('emotion-overlay');
    const emotionImage = document.getElementById('emotion-image');
    const body = document.body;

    // Ustaw emotkę jako treść tekstową
    emotionImage.textContent = emotionUnicode[emotion] || "❓";

    // 1. Przyciemnij stronę
    body.classList.add('dimmed');

    // 2. Po chwili wyświetl emotkę
    setTimeout(() => {
        overlay.style.display = 'flex'; // Pokaż emotkę
    }, 500);

    // 3. Po 2,5 sekundy ukryj emotkę i rozjaśnij stronę
    setTimeout(() => {
        overlay.style.display = 'none'; // Ukryj emotkę
        body.classList.remove('dimmed'); // Rozjaśnij stronę
        body.classList.add('normal'); // Powrót do normalności
    }, 3000);

    // 4. Po dodatkowym czasie usuń klasę "normal" dla powrotu
    setTimeout(() => {
        body.classList.remove('normal');
    }, 3000);
}
