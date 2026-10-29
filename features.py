"""Извлечение признаков — ровно то же, что в samples.py при обучении.

Если меняешь параметры здесь, их надо менять и в samples.py (и переобучать
модель), иначе модель получит не те признаки, на которых училась.
Лучше всего сделать в samples.py `from features import extract_features, ...`,
чтобы код был в одном месте.
"""
import numpy as np
import librosa

SR = 22050
HOP_LENGTH = 512
MAX_TIME_STEPS = 130          # 130 кадров * 512 / 22050 ≈ 3,02 с

USE_BASS = True               # должно совпадать с тем, на чём обучена модель
BASS_FMIN = librosa.note_to_hz('E2')
BASS_N_OCTAVES = 1

# Как считать основную хромаграмму:
#   'stft' — как в статье (кратковременное преобразование Фурье, окно 2048);
#   'cqt'  — преобразование с постоянной добротностью: в нижнем и среднем регистре
#            различает соседние полутоны (G3 и G#3 отстоят всего на ~12 Гц, а бин STFT ≈ 10,8 Гц).
# Модели из статьи обучены с 'stft' — для них здесь должно стоять 'stft'.
MAIN_CHROMA = 'cqt'

N_FEATURES = 24 if USE_BASS else 12
CLIP_SAMPLES = MAX_TIME_STEPS * HOP_LENGTH   # сколько отсчётов покрывают 130 кадров


def fit_length(features, max_steps=MAX_TIME_STEPS):
    """Обрезает или дополняет нулями матрицу (n_признаков, n_кадров) до max_steps кадров."""
    if features.shape[1] > max_steps:
        return features[:, :max_steps]
    pad_width = max_steps - features.shape[1]
    return np.pad(features, ((0, 0), (0, pad_width)), mode='constant')


def extract_features(audio, sr=SR):
    """audio — моно-сигнал с частотой sr. Возвращает матрицу (MAX_TIME_STEPS, N_FEATURES)."""
    if MAIN_CHROMA == 'cqt':
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=HOP_LENGTH)
    else:
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=HOP_LENGTH)
    chroma = fit_length(chroma)
    if not USE_BASS:
        return chroma.T

    bass = librosa.feature.chroma_cqt(
        y=audio, sr=sr, hop_length=HOP_LENGTH,
        fmin=BASS_FMIN, n_octaves=BASS_N_OCTAVES,
    )
    bass = fit_length(bass)
    return np.vstack([chroma, bass]).T