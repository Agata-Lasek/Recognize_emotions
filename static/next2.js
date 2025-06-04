const textDisplay = document.getElementById('text-display');
const emotionDisplay = document.getElementById('emotion-display');
const recordButton = document.getElementById('record-button');
const stopButton = document.getElementById('stop-button');

let mediaRecorder;
let audioChunks = [];

// Rozpocznij nagrywanie
recordButton.addEventListener('click', async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        const audioArrayBuffer = await audioBlob.arrayBuffer();

        const audioContext = new AudioContext();
        const audioBuffer = await audioContext.decodeAudioData(audioArrayBuffer);

        const wavBlob = audioBufferToWavBlob(audioBuffer);
        await uploadAndProcessAudio(wavBlob); // Wyślij nagranie do serwera i przetwórz
    };

    mediaRecorder.start();
    recordButton.style.display = 'none';
    stopButton.style.display = 'block';
});

// Zatrzymaj nagrywanie
stopButton.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        stopButton.style.display = 'none';
        recordButton.style.display = 'block';
    }
});

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

// Wyślij plik audio do serwera i przetwórz
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
            textDisplay.textContent = `Error: ${data.error}`;
            emotionDisplay.textContent = '';
        } else {
            // Wyświetl transkrypcję
            textDisplay.textContent = data.transcription || 'No text recognized';

            // Wyświetl emocje-tekst (z tex_rec.py)
            const textEmotionHTML = Object.entries(data.tex_rec_emotions || {}).map(
                ([emotion, value]) => `${emotion}: ${value.toFixed(2)}%`
            ).join('<br>');

            // Wyświetl emocje-dźwięk (z new_file_emotion.py)
            const audioEmotionHTML = Object.entries(data.new_file_emotion.probabilities || {}).map(
                ([emotion, value]) => `${emotion}: ${value.toFixed(2)}%`
            ).join('<br>');

            // Łączymy wyniki w polu wyświetlania
            emotionDisplay.innerHTML = `
                <strong>Emocje-tekst (tex_rec.py):</strong><br>${textEmotionHTML}<br><br>
                <strong>Emocje-dźwięk (new_file_emotion.py):</strong><br>${audioEmotionHTML}
            `;
        }
    } catch (error) {
        textDisplay.textContent = `Processing failed: ${error.message}`;
        emotionDisplay.textContent = '';
        console.error('Error processing audio:', error);
    }
}



