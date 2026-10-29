"""Подготовка GuitarSet для обучения модели распознавания аккордов.

Берёт аккомпанементные записи GuitarSet (comp), нарезает их по разметке аккордов,
оставляет фрагменты 36 классов (12 мажорных, 12 минорных, 12 уменьшённых трезвучий) и сохраняет их как отдельные WAV
в той же структуре, что основной датасет: папка = имя аккорда.

    guitarset_clips/
        train/Am/*.wav, train/C/*.wav, ...
        test/Am/*.wav, ...            <- записи одного гитариста, которые модель не увидит

Запуск (путь — папка, куда распакован GuitarSet с Kaggle):
    python prepare_guitarset.py C:\\datasets\\guitarset --out C:\\datasets\\guitarset_clips

Полезные ключи:
    --sevenths        брать также септаккорды, обращения и аккорды с добавленными ступенями,
                      относя их к аккорду по основному тону (G7 -> G, Am7 -> Am, Bb/D -> Bb).
                      По умолчанию берутся только чистые трезвучия в основном виде.
    --chords leadsheet  брать аккорды из партитуры, а не определённые по сыгранным нотам
    --audio mix       брать запись со звукоснимателя вместо микрофона
    --test-player 05  какого гитариста отложить для проверки (по умолчанию 05; none — никого)
    --dry-run         только посчитать, сколько фрагментов получится, ничего не записывать
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

SR = 22050                 # частота, с которой работает модель
CLIP_SEC = 130 * 512 / SR  # ≈ 3,02 с — длина входа модели
MIN_SEC = 0.75             # более короткие участки аккорда не берём
PRE_ROLL_SEC = 0.03        # чуть звука до отметки начала аккорда, чтобы не срезать удар

# Имена классов — те же, что в transpose.py и train_36.py: 12 основных тонов × мажор/минор/уменьшённый
NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
SUFFIX = {'maj': '', 'min': 'm', 'dim': 'dim'}
CLASSES = [n + SUFFIX[q] for q in ('maj', 'min', 'dim') for n in NOTE_NAMES]

PITCH = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

TRIAD = {'maj': 'maj', 'min': 'min', 'dim': 'dim'}
SEVENTH = {'7': 'maj', 'maj7': 'maj', 'maj6': 'maj', '6': 'maj', '9': 'maj', 'maj9': 'maj',
           'min7': 'min', 'min6': 'min', 'min9': 'min', 'minmaj7': 'min',
           'dim7': 'dim', 'hdim7': 'dim'}


def root_pc(root):
    """'Bb' -> 10, 'A#' -> 10, 'Eb' -> 3."""
    pc = PITCH[root[0].upper()]
    for acc in root[1:]:
        pc += 1 if acc == '#' else -1 if acc == 'b' else 0
    return pc % 12


def map_label(label, sevenths=False):
    """Метка JAMS (например 'A:min7', 'Bb:maj', 'G:7/B') -> наш класс или None."""
    if label in ('N', 'X') or ':' not in label:
        return None
    root, full_quality = label.split(':', 1)
    if full_quality.endswith('/1'):
        # В нотации Harte '/1' означает «в басу основной тон», т.е. обычное (не обращённое) звучание
        full_quality = full_quality[:-2]
    quality = re.split(r'[/(]', full_quality)[0]     # убрать бас (/B) и добавленные ступени
    if not sevenths and quality != full_quality:
        # По умолчанию берём только чистые трезвучия в основном виде: обращения (Bb/D)
        # меняют басовый тон, а добавленные ступени — хроматический профиль.
        return None
    table = {**TRIAD, **SEVENTH} if sevenths else TRIAD
    if quality not in table:
        return None
    try:
        return NOTE_NAMES[root_pc(root)] + SUFFIX[table[quality]]
    except KeyError:
        return None


def load_chords(jams_path, source):
    """Аккорды из JAMS: source='performed' — определённые по сыгранным нотам,
    'leadsheet' — из партитуры. В GuitarSet это 2-я и 1-я аннотации chord соответственно."""
    with open(jams_path, encoding='utf8') as f:
        data = json.load(f)
    chords = [a for a in data['annotations'] if a.get('namespace') == 'chord']
    if not chords:
        return []
    ann = chords[1] if source == 'performed' and len(chords) > 1 else chords[0]
    return [(float(e['time']), float(e['duration']), str(e['value'])) for e in ann['data']]


def windows(start, dur):
    """Окна длиной до CLIP_SEC внутри одного аккорда: первое от начала аккорда,
    следующие — подряд, пока остаток не короче MIN_SEC. В следующий аккорд не заходим."""
    out, t, end = [], start, start + dur
    while end - t >= MIN_SEC:
        out.append((max(0.0, t - PRE_ROLL_SEC), min(t + CLIP_SEC, end)))
        t += CLIP_SEC
    return out


def find_files(root):
    jams, audio = {}, {}
    for d, _, files in os.walk(root):
        for name in files:
            path = os.path.join(d, name)
            if name.endswith('.jams'):
                jams[name[:-5]] = path
            elif name.endswith('.wav'):
                audio[name[:-4]] = path
    return jams, audio


def read_audio(path):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def write_audio(path, y):
    import soundfile as sf
    sf.write(path, y, SR)


def main():
    ap = argparse.ArgumentParser(description='Нарезка GuitarSet на фрагменты аккордов')
    ap.add_argument('src', help='папка с распакованным GuitarSet')
    ap.add_argument('--out', default='guitarset_clips', help='куда сохранить фрагменты')
    ap.add_argument('--audio', choices=['mic', 'mix'], default='mic')
    ap.add_argument('--chords', choices=['performed', 'leadsheet'], default='performed')
    ap.add_argument('--sevenths', action='store_true')
    ap.add_argument('--test-player', default='05')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    jams, audio = find_files(args.src)
    comp = sorted(k for k in jams if k.endswith('_comp'))
    print(f'Найдено файлов разметки: {len(jams)}, из них аккомпанемент (comp): {len(comp)}')
    if not comp:
        raise SystemExit('Не нашёл файлов *_comp.jams — проверь путь к GuitarSet.')

    counts = defaultdict(Counter)       # split -> Counter(класс)
    skipped = Counter()                 # исходные метки, которые не вошли
    missing_audio = 0

    for track in comp:
        audio_path = audio.get(f'{track}_{args.audio}')
        if audio_path is None:
            missing_audio += 1
            continue
        split = 'test' if track.split('_')[0] == args.test_player else 'train'
        y = None
        for i, (t, d, label) in enumerate(load_chords(jams[track], args.chords)):
            cls = map_label(label, args.sevenths)
            if cls is None:
                skipped[label] += 1
                continue
            for k, (a, b) in enumerate(windows(t, d)):
                counts[split][cls] += 1
                if args.dry_run:
                    continue
                if y is None:
                    y = read_audio(audio_path)
                clip = y[int(a * SR):int(b * SR)]
                folder = os.path.join(args.out, split, cls)
                os.makedirs(folder, exist_ok=True)
                write_audio(os.path.join(folder, f'{track}_{i:03d}_{k}.wav'), clip)

    if missing_audio:
        print(f'Нет аудио «{args.audio}» для {missing_audio} записей — '
              f'проверь, что в папке есть файлы *_{args.audio}.wav')
    for split in ('train', 'test'):
        c = counts[split]
        print(f'\n{split}: всего {sum(c.values())} фрагментов')
        for q in ('maj', 'min', 'dim'):
            row = [n + SUFFIX[q] for n in NOTE_NAMES]
            print('  ' + '  '.join(f'{r}:{c[r]}' for r in row))
    print('\nСамые частые метки, которые не вошли:',
          ', '.join(f'{l} ({n})' for l, n in skipped.most_common(10)))
    if args.dry_run:
        print('\n(--dry-run: файлы не записывались)')


if __name__ == '__main__':
    main()