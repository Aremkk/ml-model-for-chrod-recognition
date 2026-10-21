from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from samples import le, MAX_TIME_STEPS, N_FEATURES
import tensorflow as tf

SEED = 15
tf.keras.utils.set_random_seed(SEED)


def build_model(num_classes, n_features=N_FEATURES):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(MAX_TIME_STEPS, n_features)))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


print("\nОбучаем LSTM-модель")
num_classes = len(le.classes_)
model = build_model(num_classes)


print("\nИнформация о модели")
model.summary()
