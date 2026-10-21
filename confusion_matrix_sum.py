import numpy as np

SEEDS = [11, 12, 13, 14, 15]
for tag in ('chroma', 'bass'):
    total = sum(np.load(f'cm/{tag}_seed{s}.npy') for s in SEEDS)
    errors = total.sum() - np.trace(total)
    print(f'{tag}: ошибок {errors}')
    print(total)
    np.save(f'cm/{tag}_sum.npy', total)