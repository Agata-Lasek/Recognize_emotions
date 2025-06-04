from flask import Flask, render_template, request, jsonify
import os
from tex_rec import transcribe_audio, analyze_emotions
from new_file_emotion import predict_emotion

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'test_samples')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def normalize_emotion_names(emotions):
    """
    Zmienia nazwy emocji na spójne:
    - anger -> angry
    - happiness -> happy
    - sadness -> sad
    Usuwa emocję 'disgust', jeśli występuje.
    """
    normalized = {}
    mapping = {
        "anger": "angry",
        "happiness": "happy",
        "sadness": "sad"
    }
    for emotion, value in emotions.items():
        if emotion == "disgust":
            continue  # Pomijamy emocję 'disgust'
        normalized_emotion = mapping.get(emotion, emotion)  # Domyślnie bez zmian
        normalized[normalized_emotion] = value
    return normalized



def calculate_final_emotions(tex_emotions, audio_emotions):
    """
    Oblicz ostateczne emocje na podstawie wagi:
    0.1 * emocje z tekstu + 0.9 * emocje z mowy
    """
    final_emotions = {}
    for emotion in set(tex_emotions.keys()).intersection(audio_emotions.keys()):
        final_emotions[emotion] = 0.1 * tex_emotions.get(emotion, 0) + 0.9 * audio_emotions.get(emotion, 0)
    recognized_emotion = max(final_emotions, key=final_emotions.get, default=None)  # Najwyższa wartość
    return final_emotions, recognized_emotion


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/next')
def next_page():
    return render_template('next.html')


@app.route('/process_audio_combined', methods=['POST'])
def process_audio_combined():
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio_data']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Zapisz plik
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
        audio_file.save(file_path)

        # Przetwarzanie za pomocą tex_rec.py
        transcription = transcribe_audio(file_path)
        tex_rec_emotions = analyze_emotions(transcription) if transcription else {}

        # Normalizuj nazwy emocji w tekście
        tex_rec_emotions = normalize_emotion_names(tex_rec_emotions)

        # Przetwarzanie za pomocą new_file_emotion.py
        audio_emotion_result = predict_emotion(file_path)
        audio_emotions = normalize_emotion_names(audio_emotion_result.get('probabilities', {}))

        # Oblicz emocje ostateczne
        final_emotions, recognized_emotion = calculate_final_emotions(tex_rec_emotions, audio_emotions)

        # Zwróć dane w formacie JSON
        return jsonify({
            'transcription': transcription or 'No transcription available',
            'tex_rec_emotions': tex_rec_emotions or {},
            'audio_emotions': audio_emotions or {},
            'final_emotions': final_emotions or {},
            'recognized_emotion': recognized_emotion or 'No emotion detected'
        }), 200

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
