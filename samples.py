import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder

TRAIN_PATH = r'C:\datasets\carchive\Chords\Training'
TEST_PATH = r'C:\datasets\carchive\Chords\Test'

SR = 22050
HOP_LENGTH = 512
MAX_TIME_STEPS = 130
USE_BASS =True

BASS_FMIN = librosa.note_to_hz('E2')
BASS_N_OCTAVES = 1
N_FEATURES = 24 if USE_BASS else 12
CACHE_FILE = f'features_{"bass" if USE_BASS else "chroma"}.npz'

def fit_length(features, max_steps=MAX_TIME_STEPS):
    if features.shape[1] > max_steps:
        return features[:, :max_steps]
    pad_width = max_steps - features.shape[1]
    return np.pad(features, ((0, 0), (0, pad_width)), mode='constant')

def extract_features(audio, sr):
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=HOP_LENGTH)
    chroma = fit_length(chroma)

    if not USE_BASS:
        return chroma.T
    bass = librosa.feature.chroma_cqt(
        y=audio, sr=sr,
        hop_length=HOP_LENGTH,
        fmin=BASS_FMIN,
        n_octaves=BASS_N_OCTAVES,
    )
    bass = fit_length(bass)
    return np.vstack([chroma, bass]).T

def extract_features_from_dir(dataset_dir):
    X, y = [], []
    for chord_name in os.listdir(dataset_dir):
        chord_dir = os.path.join(dataset_dir, chord_name)

        if not os.path.isdir(chord_dir):
            continue
        for file_name in os.listdir(chord_dir):
            if file_name.endswith(('.wav', '.mp3', '.ogg')):
                file_path = os.path.join(chord_dir, file_name)
                try:
                    audio, sr = librosa.load(file_path, sr=22050)
                    X.append(extract_features(audio, sr))
                    y.append(chord_name)
                except Exception as e:
                    print(f"Ошибка при обработке {file_path}: {e}")
    return np.array(X, dtype=np.float32), np.array(y)


if os.path.exists(CACHE_FILE):
    print(f"Загружаем признаки из кэша {CACHE_FILE}")
    data = np.load(CACHE_FILE)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
else:
    print("Обрабатываем тренировочную выборку")
    X_train, y_train = extract_features_from_dir(TRAIN_PATH)

    print("Обрабатываем тестовую выборку")
    X_test, y_test = extract_features_from_dir(TEST_PATH)

    np.savez_compressed(CACHE_FILE, X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)
    print(f"Признаки сохранены в {CACHE_FILE}")

print(f"Размерности: Train X={X_train.shape}, Test X={X_test.shape}")

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)