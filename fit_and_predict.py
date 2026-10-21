from for_train_test import model, SEED
from samples import  X_train, X_test, y_train_encoded, y_test_encoded, le, USE_BASS
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import os

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=3,
    factor=0.5,
    verbose=1,
    min_delta=0.0001,
    min_lr=0.0001,
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train_encoded,
    test_size=0.15,
    stratify=y_train_encoded,
    random_state=SEED
)

model.fit(
    X_tr, y_tr,
    epochs=100, batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr]
)

y_pred_prob = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_prob, axis=1)
model.save(
    f'cm/Recognizing_chord_with_bass_seed{SEED}.keras', overwrite=True )

variant = f'хромаграмма + бас (SEED {SEED})' if USE_BASS else f'только хромаграмма (SEED {SEED})'
print(f"\nПризнаки: {variant}")
print(f"Точность: {accuracy_score(y_test_encoded, y_pred_classes):.3f}")
print("\nДетальный отчет:")
print(classification_report(y_test_encoded, y_pred_classes,
                            target_names=le.classes_, digits=3))

cm = confusion_matrix(y_test_encoded, y_pred_classes)
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(cmap='Blues', values_format='d')
plt.title(f'Матрица ошибок ({variant})')
plt.tight_layout()
#plt.savefig(f'(SEED {SEED})confusion_matrix_{"bass" if USE_BASS else "chroma"}.png', dpi=300)
plt.show()


#os.makedirs('cm', exist_ok=True)
#tag = 'bass' if USE_BASS else 'chroma'
#np.save(f'cm/{tag}_seed{SEED}.npy', cm)