"""Распознавание аккордов с микрофона в реальном времени.

Как работает:
  1. Микрофон читается непрерывно небольшими блоками (sounddevice).
  2. По громкости (RMS) ловится момент удара по струнам — онсет.
  3. После онсета записываются CLIP_SEC секунд (плюс немного звука до удара),
     то есть фрагмент того же вида, что записи в датасете: аккорд от начала звучания.
  4. Из фрагмента извлекаются те же признаки, что при обучении, и модель
     выдаёт аккорд и уверенность.

Запуск:
  pip install sounddevice
  python realtime_chords.py --model chord_model.keras
  python realtime_chords.py --list-devices            # список микрофонов
  python realtime_chords.py --device 3                # выбрать микрофон по номеру
  python realtime_chords.py --file путь\\к\\записи.wav  # проверка без микрофона
  python realtime_chords.py --file папка               # все записи в папке; метка = имя подпапки
  python realtime_chords.py --save my_records --label Bb   # сохранять каждый фрагмент в my_records\\Bb\\
"""
import argparse
import os
import queue
import sys
import time
from collections import deque

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')   # убрать служебные сообщения TensorFlow

import numpy as np
import librosa
import tensorflow as tf

from features import SR, MAX_TIME_STEPS, N_FEATURES, CLIP_SAMPLES, extract_features

# Порядок классов, как его задаёт LabelEncoder (сортировка имён папок).
# Если сохранишь le.classes_ в classes.npy, скрипт возьмёт порядок оттуда.
DEFAULT_CLASSES = ['Am', 'Bb', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G']

# --- Настройки детектора удара -------------------------------------------
BLOCK_SEC = 0.02        # длина блока, читаемого с микрофона (20 мс)
PRE_ROLL_SEC = 0.05     # сколько звука до удара добавить в начало фрагмента
CLIP_SEC = CLIP_SAMPLES / SR   # ≈ 3,02 с, как при обучении
ONSET_RATIO = 4.0       # удар = громкость блока выше фонового шума в 4 раза
MIN_RMS = 0.01          # и не тише этого абсолютного порога (защита от шороха)
REFRACTORY_SEC = 0.3    # пауза после распознавания, прежде чем ловить новый удар
MIN_CONFIDENCE = 0.5    # ниже этой уверенности результат помечается как сомнительный


def load_classes(path='classes.npy'):
    if os.path.exists(path):
        return [str(c) for c in np.load(path, allow_pickle=True)]
    return DEFAULT_CLASSES


def load_model(path):
    model = tf.keras.models.load_model(path, compile=False)
    expected = model.input_shape[-1]
    if expected != N_FEATURES:
        sys.exit(f'Модель ждёт {expected} признаков в кадре, а features.py даёт {N_FEATURES}. '
                 f'Проверь USE_BASS в features.py: он должен совпадать с тем, на чём обучалась модель.')
    return model


def predict(model, classes, audio, sr):
    """Принимает моно-сигнал любой частоты, возвращает (аккорд, уверенность, все вероятности)."""
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    x = extract_features(audio.astype(np.float32), SR)[np.newaxis]   # (1, 130, N_FEATURES)
    probs = model.predict(x, verbose=0)[0]
    i = int(np.argmax(probs))
    return classes[i], float(probs[i]), probs


def format_result(chord, conf, probs, classes):
    top = np.argsort(probs)[::-1][:3]
    alt = ', '.join(f'{classes[j]} {probs[j]:.0%}' for j in top[1:])
    mark = '' if conf >= MIN_CONFIDENCE else '  (не уверена)'
    return f'>>> {chord:<5} {conf:5.0%}{mark}   | дальше: {alt}'


AUDIO_EXT = ('.wav', '.mp3', '.ogg', '.flac')


def run_file(model, classes, path):
    """Один файл — печатает результат. Папка — проходит все файлы, метка берётся из имени
    подпапки (как в датасете), и печатает точность по каждому аккорду."""
    if not os.path.isdir(path):
        audio, sr = librosa.load(path, sr=SR, mono=True)
        chord, conf, probs = predict(model, classes, audio, sr)
        print(format_result(chord, conf, probs, classes))
        return

    stats = {}   # метка -> {предсказание: количество}
    for root, _, files in os.walk(path):
        for name in sorted(files):
            if not name.lower().endswith(AUDIO_EXT):
                continue
            label = os.path.basename(root)
            audio, sr = librosa.load(os.path.join(root, name), sr=SR, mono=True)
            chord, conf, _ = predict(model, classes, audio, sr)
            stats.setdefault(label, {}).setdefault(chord, 0)
            stats[label][chord] += 1
            print(f'{label:>5} -> {chord:<5} {conf:4.0%}  {name}')

    print('\nИтог по папкам:')
    total = correct = 0
    for label, preds in sorted(stats.items()):
        n = sum(preds.values())
        ok = preds.get(label, 0)
        total += n
        correct += ok
        detail = ', '.join(f'{c}: {k}' for c, k in sorted(preds.items(), key=lambda t: -t[1]))
        print(f'  {label:>5}: верно {ok}/{n}   ({detail})')
    if total and any(l in classes for l in stats):
        print(f'  Всего верно: {correct}/{total} = {correct / total:.1%}')


def save_clip(audio, sr, save_dir, label, chord):
    """Сохраняет фрагмент в save_dir/label/ с частотой 22050 Гц — в том же виде,
    что записи датасета, чтобы их потом можно было использовать для проверки и дообучения."""
    import soundfile as sf
    folder = os.path.join(save_dir, label or 'unlabeled')
    os.makedirs(folder, exist_ok=True)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    name = time.strftime('%Y%m%d_%H%M%S') + f'_pred-{chord}.wav'
    sf.write(os.path.join(folder, name), audio, SR)
    print(f'  сохранено: {os.path.join(folder, name)}')


def run_microphone(model, classes, device, save_dir=None, label=None):
    import sounddevice as sd

    info = sd.query_devices(device, 'input')
    sr = int(info['default_samplerate'])          # родная частота микрофона; потом ресемплинг до 22050
    block = int(BLOCK_SEC * sr)
    print(f'Микрофон: {info["name"]} ({sr} Гц)')

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata[:, 0].copy())                # берём первый канал

    pre_roll = deque(maxlen=max(1, int(PRE_ROLL_SEC / BLOCK_SEC)))
    noise = None              # оценка фонового шума (RMS)
    recording = None          # список блоков текущего фрагмента
    clip = int(CLIP_SEC * sr)  # сколько отсчётов записать после удара
    target = 0                # итоговая длина фрагмента: звук до удара + clip
    wait_until = 0.0

    # прогрев модели, чтобы первое предсказание не тормозило
    model.predict(np.zeros((1, MAX_TIME_STEPS, N_FEATURES), np.float32), verbose=0)

    with sd.InputStream(device=device, channels=1, samplerate=sr, blocksize=block,
                        dtype='float32', callback=callback):
        print('Слушаю. Сыграй аккорд (Ctrl+C — выход)...')
        while True:
            chunk = q.get()
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if recording is not None:
                recording.append(chunk)
                if sum(len(c) for c in recording) >= target:
                    audio = np.concatenate(recording)[:target]
                    chord, conf, probs = predict(model, classes, audio, sr)
                    print(format_result(chord, conf, probs, classes))
                    if save_dir:
                        save_clip(audio, sr, save_dir, label, chord)
                    recording = None
                    wait_until = time.time() + REFRACTORY_SEC
                    print('Слушаю...')
                continue

            if noise is None:
                noise = rms
            is_onset = (time.time() >= wait_until and rms > MIN_RMS and rms > ONSET_RATIO * noise)
            if is_onset:
                recording = list(pre_roll) + [chunk]
                target = sum(len(c) for c in pre_roll) + clip
                print('  ...удар, записываю')
            else:
                noise = 0.95 * noise + 0.05 * rms   # медленно подстраиваемся под фон
                pre_roll.append(chunk)


def main():
    ap = argparse.ArgumentParser(description='Распознавание аккордов с микрофона')
    ap.add_argument('--model', default='chord_model.keras', help='файл сохранённой модели (.keras или .h5)')
    ap.add_argument('--classes', default='classes.npy', help='файл с порядком классов (необязательно)')
    ap.add_argument('--device', type=int, default=None, help='номер микрофона (см. --list-devices)')
    ap.add_argument('--list-devices', action='store_true', help='показать аудиоустройства и выйти')
    ap.add_argument('--file', help='распознать аккорд в аудиофайле (или во всех файлах папки) вместо микрофона')
    ap.add_argument('--save', metavar='DIR', help='сохранять каждый записанный фрагмент в эту папку')
    ap.add_argument('--label', help='какой аккорд ты сейчас играешь (имя подпапки для --save)')
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    model = load_model(args.model)
    classes = load_classes(args.classes)
    if args.file:
        run_file(model, classes, args.file)
    else:
        try:
            run_microphone(model, classes, args.device, args.save, args.label)
        except KeyboardInterrupt:
            print('\nГотово.')


if __name__ == '__main__':
    main()