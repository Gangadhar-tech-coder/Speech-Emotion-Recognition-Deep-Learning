import os
import pickle
import numpy as np
import librosa
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from pydub import AudioSegment

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# ============================================================
# 📌 MODEL PATH — UPDATE THESE 3 PATHS WITH YOUR MODEL FILES
# ============================================================
MODEL_PATH         = 'model/emotion_model.h5'       # ← Path to your .h5 model
LABEL_ENCODER_PATH = 'model/label_encoder.pkl'      # ← Path to label_encoder.pkl
SCALER_PATH        = 'model/scaler.pkl'             # ← Path to scaler.pkl
# ============================================================

# Emotion emoji mapping
EMOTION_EMOJI = {
    'neutral'  : '😐',
    'calm'     : '😌',
    'happy'    : '😊',
    'sad'      : '😢',
    'angry'    : '😠',
    'fear'     : '😨',
    'disgust'  : '🤢',
    'surprise' : '😲',
    'ps'       : '😲',
}

EMOTION_COLOR = {
    'neutral'  : '#7f8c8d',
    'calm'     : '#3498db',
    'happy'    : '#f1c40f',
    'sad'      : '#2980b9',
    'angry'    : '#e74c3c',
    'fear'     : '#9b59b6',
    'disgust'  : '#27ae60',
    'surprise' : '#e67e22',
    'ps'       : '#e67e22',
}

# Load model files
print('🔄 Loading model...')
try:
    model         = load_model(MODEL_PATH)
    label_encoder = pickle.load(open(LABEL_ENCODER_PATH, 'rb'))
    scaler        = pickle.load(open(SCALER_PATH, 'rb'))
    print('✅ Model loaded successfully!')
except Exception as e:
    print(f'❌ Error loading model: {e}')
    model = label_encoder = scaler = None


def extract_features(file_path):
    """Extract MFCC + Chroma + MEL features from audio file"""
    try:
        # Convert audio to WAV if needed
        if not file_path.endswith(".wav"):
            sound = AudioSegment.from_file(file_path)
            wav_path = file_path + ".wav"
            sound.export(wav_path, format="wav")
            file_path = wav_path

        audio, sr = librosa.load(
            file_path,
            res_type='kaiser_fast',
            duration=3,
            sr=22050,
            offset=0.5
        )

        features = []

        # MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfccs.T, axis=0))

        # Chroma
        stft = np.abs(librosa.stft(audio))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        features.extend(np.mean(chroma.T, axis=0))

        # MEL
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        features.extend(np.mean(mel.T, axis=0))

        return np.array(features)

    except Exception as e:
        print(f'Feature extraction error: {e}')
        return None

def predict_emotion(file_path):
    """Predict emotion from audio file"""
    if model is None:
        return None, None, None

    features = extract_features(file_path)
    if features is None:
        return None, None, None

    features_scaled = scaler.transform([features])

    # Reshape for CNN
    features_reshaped = features_scaled.reshape(
        features_scaled.shape[0], features_scaled.shape[1], 1
    )

    predictions  = model.predict(features_reshaped, verbose=0)[0]
    emotion_idx  = np.argmax(predictions)
    emotion      = label_encoder.classes_[emotion_idx]
    confidence   = float(predictions[emotion_idx]) * 100

    # All emotion probabilities
    all_emotions = {
        label_encoder.classes_[i]: round(float(predictions[i]) * 100, 1)
        for i in range(len(predictions))
    }

    return emotion, confidence, all_emotions


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        filename  = secure_filename(file.filename)
        filepath  = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        emotion, confidence, all_emotions = predict_emotion(filepath)

        # Clean up uploaded file
        os.remove(filepath)

        if emotion is None:
            return jsonify({'error': 'Could not process audio file'}), 500

        return jsonify({
            'emotion'     : emotion,
            'confidence'  : round(confidence, 1),
            'emoji'       : EMOTION_EMOJI.get(emotion, '🎤'),
            'color'       : EMOTION_COLOR.get(emotion, '#3498db'),
            'all_emotions': all_emotions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True, port=5000)
