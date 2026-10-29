"""Обучение модели распознавания аккордов на 36 классах:
12 мажорных, 12 минорных и 12 уменьшённых трезвучий.

Отдельный скрипт: эксперимент из статьи (8 классов, samples.py / fit_and_predict.py)
он не трогает.

Что делает:
  1. Читает обучающие данные из нескольких папок (основной датасет + GuitarSet + ...).
     Метка = имя подпапки, имена как в CLASSES ниже (Bb, а не A#; F#m, Ebdim и т.д.).
  2. Делит обучающие данные на обучение и валидацию (валидацию НЕ транспонирует).
  3. Обучает с транспонированием на лету: каждый пример в каждой эпохе сдвигается
     на случайное число полутонов 0..11, метка пересчитывается.
  4. Проверяет модель на нескольких тестовых наборах отдельно и сохраняет
     модель, classes.npy и матрицы ошибок.

Запуск:
    python train_36.py --seed 11
"""
import argparse
import hashlib
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from features import SR, MAX_TIME_STEPS, N_FEATURES, USE_BASS, MAIN_CHROMA, extract_features

# ---------------------------------------------------------------- настройки
TRAIN_DIRS = [
    r'C:\datasets\carchive\Chords\Training',
    r'C:\datasets\guitarset_clips\train',
    # r'C:\...\my_records_train',      # свои записи, когда будут
]
TEST_DIRS = {
    'исходный тест (8 аккордов)': r'C:\datasets\carchive\Chords\Test',
    'GuitarSet, гитарист 05': r'C:\datasets\guitarset_clips\test',
    # 'мои записи': r'C:\...\my_records_test',
}
CACHE_DIR = 'cache36'
OUT_DIR = 'models36'

NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
SUFFIX = {'maj': '', 'min': 'm', 'dim': 'dim'}
QUALITIES = ('maj', 'min', 'dim')
CLASSES = [n + SUFFIX[q] for q in QUALITIES for n in NOTE_NAMES]   # 36 классов
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
AUDIO_EXT = ('.wav', '.mp3', '.ogg', '.flac')

# Таблица транспонирования: SHIFT[i, k] = номер класса i, сдвинутого на k полутонов вверх.
# Благодаря порядку CLASSES это просто сдвиг основного тона внутри своей «дюжины».
SHIFT = np.array([[(i // 12) * 12 + (i % 12 + k) % 12 for k in range(12)]
                  for i in range(len(CLASSES))], dtype=np.int32)


# ---------------------------------------------------------------- данные
def load_dir(path):
    """Признаки и метки из папки вида path/<аккорд>/*.wav, с кэшем на диске."""
    key = hashlib.md5(f'{os.path.abspath(path)}|{USE_BASS}|{MAIN_CHROMA}'.encode()).hexdigest()[:10]
    cache = os.path.join(CACHE_DIR, f'{os.path.basename(os.path.normpath(path))}_{key}.npz')
    if os.path.exists(cache):
        d = np.load(cache)
        return d['X'], d['y']

    X, y, unknown = [], [], set()
    for label in sorted(os.listdir(path)):
        folder = os.path.join(path, label)
        if not os.path.isdir(folder):
            continue
        if label not in CLASS_INDEX:
            unknown.add(label)
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(AUDIO_EXT):
                try:
                    audio, _ = librosa.load(os.path.join(folder, name), sr=SR, mono=True)
                    X.append(extract_features(audio, SR))
                    y.append(CLASS_INDEX[label])
                except Exception as e:
                    print(f'  ошибка в {name}: {e}')
    if unknown:
        print(f'  ВНИМАНИЕ: папки с неизвестными именами пропущены: {sorted(unknown)}. '
              f'Имена должны быть как в CLASSES (например Bb, а не A#).')
    X = np.array(X, dtype=np.float32).reshape(-1, MAX_TIME_STEPS, N_FEATURES)
    y = np.array(y, dtype=np.int32)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def describe(y, title):
    counts = np.bincount(y, minlength=len(CLASSES))
    print(f'{title}: {len(y)} примеров')
    for qi, q in enumerate(QUALITIES):
        row = counts[qi * 12:(qi + 1) * 12]
        print('   ' + '  '.join(f'{CLASSES[qi * 12 + j]}:{row[j]}' for j in range(12)))


# ---------------------------------------------------------------- транспонирование
def make_train_dataset(X, y, w, batch_size, seed):
    shift_table = tf.constant(SHIFT)
    n_blocks = N_FEATURES // 12          # 1 (только хромаграмма) или 2 (+ басовая)

    def transpose(x, label, weight):
        k = tf.random.uniform([], 0, 12, dtype=tf.int32)
        parts = [tf.roll(x[:, b * 12:(b + 1) * 12], shift=k, axis=1) for b in range(n_blocks)]
        x = tf.concat(parts, axis=1)
        return x, tf.gather_nd(shift_table, tf.stack([label, k])), weight

    ds = tf.data.Dataset.from_tensor_slices((X, y, w))
    ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(transpose, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def quality_weights(y):
    """Вес примера. При случайном сдвиге каждый пример равновероятно становится
    любым из 12 аккордов своего типа, поэтому выравнивать нужно только типы:
    уменьшённых в данных обычно намного меньше, чем мажорных и минорных."""
    q = y // 12
    counts = np.bincount(q, minlength=3).astype(np.float64)
    present = counts > 0
    w_q = np.zeros(3)
    w_q[present] = len(y) / (present.sum() * counts[present])
    return w_q[q].astype(np.float32)


# ---------------------------------------------------------------- модель
def build_model(units):
    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=0.0, input_shape=(MAX_TIME_STEPS, N_FEATURES)),
        tf.keras.layers.LSTM(units, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(units // 2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate(model, X, y, title, tag):
    pred = np.argmax(model.predict(X, verbose=0), axis=1)
    present = sorted(set(y.tolist()))
    acc = accuracy_score(y, pred)
    print(f'\n=== {title}: точность {acc:.3f} ({(pred == y).sum()}/{len(y)}) ===')
    print(classification_report(y, pred, labels=present,
                                target_names=[CLASSES[i] for i in present], digits=3, zero_division=0))
    # частые ошибки
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    off = [(cm[i, j], CLASSES[i], CLASSES[j]) for i in range(len(CLASSES))
           for j in range(len(CLASSES)) if i != j and cm[i, j] > 0]
    top = sorted(off, reverse=True)[:8]
    if top:
        print('Частые ошибки: ' + ', '.join(f'{a}→{b} ({n})' for n, a, b in top))
    np.save(os.path.join(OUT_DIR, f'cm_{tag}.npy'), cm)
    return acc


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--units', type=int, default=128, help='нейронов в первом LSTM (во втором вдвое меньше)')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--no-transpose', action='store_true', help='обучить без транспонирования (для сравнения)')
    ap.add_argument('--lr-patience', type=int, default=10, help='эпох без улучшения до снижения скорости обучения')
    ap.add_argument('--stop-patience', type=int, default=25, help='эпох без улучшения до ранней остановки')
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    print(f'Признаки: основная хромаграмма {MAIN_CHROMA}, басовая {"да" if USE_BASS else "нет"}')
    os.makedirs(OUT_DIR, exist_ok=True)

    parts = []
    for d in TRAIN_DIRS:
        print(f'Загружаю обучающие данные: {d}')
        parts.append(load_dir(d))
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    describe(y, '\nОбучающие данные до разбиения')

    try:
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=args.seed)
    except ValueError:      # в каком-то классе меньше 2 примеров — делим без стратификации
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=args.seed)

    if args.no_transpose:
        train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).shuffle(len(X_tr), seed=args.seed) \
            .batch(args.batch).prefetch(tf.data.AUTOTUNE)
    else:
        train_ds = make_train_dataset(X_tr, y_tr, quality_weights(y_tr), args.batch, args.seed)

    model = build_model(args.units)
    model.summary()
    model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.stop_patience, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=args.lr_patience,
                                                 min_delta=1e-4, min_lr=1e-5, verbose=1),
        ],
    )

    tag = f'36cls_{MAIN_CHROMA}_{"bass" if USE_BASS else "nobass"}_{"notr" if args.no_transpose else "tr"}_u{args.units}_seed{args.seed}'
    model.save(os.path.join(OUT_DIR, f'model_{tag}.keras'))
    np.save(os.path.join(OUT_DIR, 'classes.npy'), np.array(CLASSES))
    print(f'\nМодель сохранена: {OUT_DIR}/model_{tag}.keras, порядок классов: {OUT_DIR}/classes.npy')

    evaluate(model, X_val, y_val, 'валидация (без транспонирования)', f'{tag}_val')
    for n, (title, d) in enumerate(TEST_DIRS.items(), 1):
        if not os.path.isdir(d):
            print(f'\n(пропускаю «{title}»: нет папки {d})')
            continue
        Xt, yt = load_dir(d)
        evaluate(model, Xt, yt, title, f'{tag}_test{n}')


if __name__ == '__main__':
    main()