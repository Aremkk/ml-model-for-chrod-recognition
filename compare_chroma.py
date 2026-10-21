"""Сравнение средних хроматических профилей нескольких групп записей.

Показывает, что модель «видит» в аккорде: среднюю энергию по 12 классам высоты
в основной хромаграмме и в басовой. Помогает понять, почему твой Bb похож для
модели на Bdim: сравни свои записи Bb с Bb и Bdim из датасета.

Пример:
  python compare_chroma.py ^
      "мой Bb=my_records\\Bb" ^
      "датасет Bb=C:\\datasets\\carchive\\Chords\\Test\\Bb" ^
      "датасет Bdim=C:\\datasets\\carchive\\Chords\\Test\\Bdim"

Каждый аргумент — «подпись=папка». Результат: таблица в консоли и картинка chroma_compare.png.
"""
import os
import sys

import numpy as np
import librosa
import matplotlib.pyplot as plt

from features import SR, USE_BASS, extract_features

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']   # A# = B♭
AUDIO_EXT = ('.wav', '.mp3', '.ogg', '.flac')


def mean_profile(folder, limit=40):
    """Средний профиль по всем кадрам звучания (кадры дополнения нулями не учитываются)."""
    frames = []
    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(AUDIO_EXT)][:limit]
    for name in files:
        audio, _ = librosa.load(os.path.join(folder, name), sr=SR, mono=True)
        x = extract_features(audio, SR)
        frames.append(x[np.any(x != 0, axis=1)])
    if not frames:
        sys.exit(f'В папке {folder} нет аудиофайлов')
    return np.concatenate(frames).mean(axis=0), len(files)


def main(args):
    if not args:
        print(__doc__)
        return
    groups = []
    for a in args:
        title, folder = a.split('=', 1)
        prof, n = mean_profile(folder)
        groups.append((f'{title} (n={n})', prof))

    parts = [('Основная хромаграмма', slice(0, 12))]
    if USE_BASS:
        parts.append(('Басовая хромаграмма (E2–E3)', slice(12, 24)))

    # таблица в консоли
    for part_title, sl in parts:
        print(f'\n{part_title}')
        print(' ' * 22 + ''.join(f'{n:>6}' for n in NOTES))
        for title, prof in groups:
            print(f'{title[:22]:<22}' + ''.join(f'{v:6.2f}' for v in prof[sl]))

    # картинка
    fig, axs = plt.subplots(1, len(parts), figsize=(6 * len(parts), 3.5), squeeze=False)
    width = 0.8 / len(groups)
    for ax, (part_title, sl) in zip(axs[0], parts):
        for k, (title, prof) in enumerate(groups):
            ax.bar(np.arange(12) + k * width - 0.4 + width / 2, prof[sl], width, label=title)
        ax.set_xticks(range(12))
        ax.set_xticklabels(NOTES)
        ax.set_title(part_title)
        ax.set_ylim(0, 1)
    axs[0][0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('chroma_compare.png', dpi=150)
    print('\nКартинка сохранена: chroma_compare.png')
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])