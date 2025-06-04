const textDisplay = document.getElementById('text-display');
const emotionDisplay = document.getElementById('emotion-display');
const recordButton = document.getElementById('record-button');
const stopButton = document.getElementById('stop-button');
const audioPlayer = document.getElementById('audio-player');
const audioSource = document.getElementById('audio-source');
const waveform = document.getElementById('waveform');

let mediaRecorder;
let audioChunks = [];
let animationFrameId;

function setAudioPlayer(filePath) {
    const audioPlayer = document.getElementById('audio-player');
    const audioSource = document.getElementById('audio-source');

    // Ustaw źródło pliku
    audioSource.src = filePath;

    // Załaduj nowe dane do odtwarzacza
    audioPlayer.load();

    // Wyświetl kontrolki odtwarzacza
    audioPlayer.style.display = 'block';
}


// Funkcja do wyświetlania pustych fal dźwiękowych
function drawIdleWaveform() {
    const canvasCtx = waveform.getContext('2d');
    let phase = 0;

    function draw() {
        animationFrameId = requestAnimationFrame(draw);

        // Tło i wymiary
        canvasCtx.fillStyle = 'black'; // Czarny kolor tła
        canvasCtx.fillRect(0, 0, waveform.width, waveform.height);

        // Parametry linii
        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = 'white'; // Użycie białego koloru jak w visualizeWaveform

        canvasCtx.beginPath();

        // Rysowanie sinusoidalnych fal
        const sliceWidth = waveform.width / 50;
        let x = 0;

        for (let i = 0; i <= 50; i++) {
            const y = waveform.height / 2 + Math.sin(phase + i * 0.5) * (waveform.height / 4); // Skala dopasowana do wysokości
            if (i === 0) {
                canvasCtx.moveTo(x, y);
            } else {
                canvasCtx.lineTo(x, y);
            }
            x += sliceWidth;
        }

        canvasCtx.lineTo(waveform.width, waveform.height / 2);
        canvasCtx.stroke();

        phase += 0.1; // Stała faza animacji
    }

    draw();
}


// Funkcja do wizualizacji fal dźwiękowych podczas nagrywania
function visualizeWaveform(stream) {
    audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();

    analyser.fftSize = 256;
    source.connect(analyser);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    const canvasCtx = waveform.getContext('2d');

    function draw() {
        animationFrameId = requestAnimationFrame(draw);
        analyser.getByteTimeDomainData(dataArray);

        canvasCtx.fillStyle = 'black';
        canvasCtx.fillRect(0, 0, waveform.width, waveform.height);

        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = 'white';

        canvasCtx.beginPath();

        const sliceWidth = waveform.width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * waveform.height) / 2;

            if (i === 0) {
                canvasCtx.moveTo(x, y);
            } else {
                canvasCtx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        canvasCtx.lineTo(waveform.width, waveform.height / 2);
        canvasCtx.stroke();
    }

    draw();
}

// Funkcja konwertująca AudioBuffer na WAV
function audioBufferToWavBlob(buffer) {
    const numOfChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const length = buffer.length * numOfChannels * 2 + 44;
    const arrayBuffer = new ArrayBuffer(length);
    const view = new DataView(arrayBuffer);

    let offset = 0;

    // Funkcje pomocnicze do pisania w nagłówku WAV
    const writeString = (str) => {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset++, str.charCodeAt(i));
        }
    };
    const writeUint16 = (value) => {
        view.setUint16(offset, value, true);
        offset += 2;
    };
    const writeUint32 = (value) => {
        view.setUint32(offset, value, true);
        offset += 4;
    };

    // Tworzenie nagłówka WAV
    writeString('RIFF'); // RIFF chunk descriptor
    writeUint32(length - 8); // File size - 8 bytes
    writeString('WAVE'); // Format
    writeString('fmt '); // Subchunk1 ID
    writeUint32(16); // Subchunk1 size (16 for PCM)
    writeUint16(1); // Audio format (1 for PCM)
    writeUint16(numOfChannels); // Number of channels
    writeUint32(sampleRate); // Sample rate
    writeUint32(sampleRate * numOfChannels * 2); // Byte rate
    writeUint16(numOfChannels * 2); // Block align
    writeUint16(16); // Bits per sample
    writeString('data'); // Subchunk2 ID
    writeUint32(buffer.length * numOfChannels * 2); // Subchunk2 size

    // Dane audio
    const interleaved = interleave(buffer);
    for (let i = 0; i < interleaved.length; i++, offset += 2) {
        const sample = Math.max(-1, Math.min(1, interleaved[i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}

// Funkcja interleavingu dla wielokanałowego audio
function interleave(buffer) {
    const numOfChannels = buffer.numberOfChannels;
    const length = buffer.length;
    const interleaved = new Float32Array(length * numOfChannels);

    let inputIndex = 0;
    for (let i = 0; i < length; i++) {
        for (let channel = 0; channel < numOfChannels; channel++) {
            interleaved[inputIndex++] = buffer.getChannelData(channel)[i];
        }
    }
    return interleaved;
}

// Obsługa nagrywania i przesyłania pliku
recordButton.addEventListener('click', async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    // Rozpocznij wizualizację dźwięku
    visualizeWaveform(stream);

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        const audioContext = new AudioContext();
        const audioBuffer = await audioContext.decodeAudioData(await audioBlob.arrayBuffer());
        const wavBlob = audioBufferToWavBlob(audioBuffer);
        await uploadAndProcessAudio(wavBlob);

        // Zatrzymaj wizualizację i wróć do fal idle
        cancelAnimationFrame(animationFrameId);
        drawIdleWaveform();
    };

    mediaRecorder.start();
    recordButton.style.display = 'none';
    stopButton.style.display = 'block';
});

stopButton.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
        stopButton.style.display = 'none';
        recordButton.style.display = 'block';
        cancelAnimationFrame(animationFrameId);
        drawIdleWaveform();
    }
});



function showEmotionReaction(emotion) {
    const overlay = document.getElementById('emotion-overlay');
    const emotionImage = document.getElementById('emotion-image');

    // Ustaw ikonę/emotkę w overlay
    const emotionIcons = {
        happy: "😀",
        sad: "😢",
        angry: "😡",
        fear: "😨",
        disgust: "🤢",
        neutral: "😐",
        surprise: "😲"
    };

    emotionImage.textContent = emotionIcons[emotion] || "❓";

    // Pokaż reakcję
    overlay.style.display = 'flex';
    setTimeout(() => {
        overlay.style.display = 'none';
    }, 2500); // 2.5 sekundy
}

const emotionResponses = {
    happy: [
        "Jesteś radosny - Ciesz się tą chwilą, to wspaniale czuć szczęście!",
        "Jesteś radosny - Twoja radość jest zaraźliwa! Podziel się nią z innymi.",
        "Jesteś radosny - Uśmiechnij się jeszcze szerzej – to Twój moment!",
        "Jesteś radosny - To idealny czas na coś, co kochasz robić.",
        "Jesteś radosny - Radość to energia – wykorzystaj ją na coś ekscytującego."
    ],
    sad: [
        "Jesteś smutny - Może ulubiona książka lub piosenka poprawi Ci nastrój?",
        "Jesteś smutny - Czasem spacer w spokojnym miejscu pomaga znaleźć ukojenie.",
        "Jesteś smutny - Spróbuj zapisać swoje myśli – to pomaga spojrzeć na nie inaczej.",
        "Jesteś smutny - Chwile smutku bywają trudne, ale zawsze prowadzą do czegoś lepszego.",
        "Jesteś smutny - Wybierz coś, co zawsze Cię rozluźnia – ciepła herbata, koc i film?"
    ],
    angry: [
        "Jesteś zły - Weź kilka głębokich oddechów – pomoże Ci to się uspokoić.",
        "Jesteś zły - Może szybki spacer lub kilka ćwiczeń pomoże pozbyć się napięcia.",
        "Jesteś zły - Posłuchaj głośnej muzyki – czasem wyrażenie złości przez dźwięk działa.",
        "Jesteś zły - Zapisz, co Cię zdenerwowało – to może dać jasny obraz sytuacji.",
        "Jesteś zły - Zrób coś, co pozwala Ci się skupić – rysowanie, układanie puzzli czy gotowanie."
    ],
    fear: [
        "Boisz się - Weź głęboki oddech i skup się na jednej małej rzeczy w pokoju – to pomaga wrócić do równowagi.",
        "Boisz się - Spróbuj przypomnieć sobie sytuacje, w których już przezwyciężyłeś strach.",
        "Boisz się - Wyobraź sobie, że sytuacja, której się boisz, już minęła – to tylko chwilowy stan.",
        "Boisz się - Spokojna muzyka lub ciepłe światło może pomóc złagodzić napięcie.",
        "Boisz się - Zrób coś małego, co odwróci uwagę – napij się wody lub przejdź kilka kroków."
    ],
    disgust: [
        "Jesteś zniesmaczony - Spróbuj pomyśleć o czymś zupełnie innym – pięknym widoku lub ulubionym smaku.",
        "Jesteś zniesmaczony - Umieść odczucie w skali – jak silne jest? To pomoże odzyskać kontrolę.",
        "Jesteś zniesmaczony - Weź głęboki oddech i przypomnij sobie coś, co sprawia Ci przyjemność.",
        "Jesteś zniesmaczony - Zmiana otoczenia lub włączenie świeżego zapachu może pomóc zapomnieć o wstręcie.",
        "Jesteś zniesmaczony - Czysta przestrzeń wokół może złagodzić ten odruch – spróbuj uporządkować coś obok siebie."
    ],
    neutral: [
        "Jesteś neutralny - Idealny moment, by odpocząć i złapać równowagę.",
        "Jesteś neutralny - Czy wiesz, że kosmos jest całkowicie cichy? Żadnego dźwięku!",
        "Jesteś neutralny - Spokój to siła. Może to czas na drobną refleksję?",
        "Jesteś neutralny - To świetny moment, by przeczytać coś inspirującego.",
        "Jesteś neutralny - Zrób coś miłego dla siebie. Nawet drobne rzeczy mają znaczenie."
    ]
};

const emotionUnicode = {
    happy: "😀",
    sad: "😢",
    angry: "😡",
    fear: "😨",
    disgust: "🤢",
    neutral: "😐",
    surprise: "😲"
};

// Funkcja do syntezowania mowy
function speakResponse(text) {
    const synth = window.speechSynthesis;

    if (!synth) {
        console.error('Speech synthesis not supported in this browser.');
        return;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'pl-PL'; // Ustaw język na polski
    utterance.pitch = 1; // Ton głosu (1 to standardowy)
    utterance.rate = 1; // Szybkość mowy (1 to standardowa)
    synth.speak(utterance);
}

// Funkcja aktualizacji odpowiedzi na podstawie emocji
function updateEmotionResponse(emotion) {
    const responseContainer = document.getElementById('emotion-response');
    const emojiContainer = document.getElementById('response-emoji');

    let randomResponse;

    if (emotionResponses[emotion]) {
        // Wybierz losową odpowiedź
        randomResponse = emotionResponses[emotion][Math.floor(Math.random() * emotionResponses[emotion].length)];
        responseContainer.textContent = randomResponse; // Wyświetl odpowiedź
    } else {
        randomResponse = "Nieznana emocja.";
        responseContainer.textContent = randomResponse;
    }

    // Ustaw emotkę
    emojiContainer.textContent = emotionUnicode[emotion] || "❓";

    // Wyłącz odczytanie odpowiedzi na głos
    // speakResponse(randomResponse); // Zakomentuj lub usuń tę linię, aby wyłączyć dźwięk
}







async function uploadAndProcessAudio(blob) {
    const formData = new FormData();
    formData.append('audio_data', blob, 'recording.wav');

    try {
        const response = await fetch('/process_audio_combined', {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();

        if (data.error) {
            // Wyświetl błąd w odpowiednich miejscach
            textDisplay.textContent = `Error: ${data.error}`;
            document.getElementById('text-emotions').textContent = 'Error occurred.';
            document.getElementById('audio-emotions').textContent = 'Error occurred.';
        } else {
            // Wyświetl transkrypcję
            textDisplay.textContent = data.transcription || 'No text recognized';

            // Wyświetl emocje z tex_rec.py (tekst)
            const textEmotionHTML = Object.entries(data.tex_rec_emotions || {}).map(
                ([emotion, value]) => `${emotion}: ${value.toFixed(2)}%`
            ).join('<br>');
            document.getElementById('text-emotions').innerHTML = textEmotionHTML;

            // Wyświetl emocje z new_file_emotion.py (dźwięk)
            const audioEmotionHTML = Object.entries(data.audio_emotions || {}).map(
                ([emotion, value]) => `${emotion}: ${value.toFixed(2)}%`
            ).join('<br>');
            document.getElementById('audio-emotions').innerHTML = audioEmotionHTML;

            // Wyświetl ostateczne emocje
            const finalEmotionHTML = Object.entries(data.final_emotions || {}).map(
                ([emotion, value]) => `${emotion}: ${value.toFixed(2)}%`
            ).join('<br>');
            document.getElementById('final-emotions').innerHTML = finalEmotionHTML;

            // Wyświetl rozpoznaną emocję
            const yourMood = document.getElementById('your-mood');
            if (data.recognized_emotion) {
                yourMood.textContent = `Recognized Emotion: ${data.recognized_emotion}`;

                // Zaktualizuj odpowiedź i odczytaj ją na głos
                updateEmotionResponse(data.recognized_emotion);
            } else {
                yourMood.textContent = `Recognized Emotion: No emotion detected`;
                speakResponse("Nie wykryto emocji.");
            }

            // Ustaw plik do odsłuchania
            setAudioPlayer(URL.createObjectURL(blob));
        }
    } catch (error) {
        // Obsłuż błąd
        textDisplay.textContent = `Processing failed: ${error.message}`;
        document.getElementById('text-emotions').textContent = 'Error occurred.';
        document.getElementById('audio-emotions').textContent = 'Error occurred.';
        console.error('Error processing audio:', error);
    }
}


// Rozpocznij wizualizację pustych fal
drawIdleWaveform();

