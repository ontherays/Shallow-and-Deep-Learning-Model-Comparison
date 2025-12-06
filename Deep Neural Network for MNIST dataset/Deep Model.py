from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

deep_model = Sequential([
    Dense(512, activation='relu', input_shape=(784,)),
    BatchNormalization(),

    Dense(256, activation='relu'),
    BatchNormalization(),

    Dense(256, activation='relu'),
    BatchNormalization(),

    Dense(128, activation='relu'),
    BatchNormalization(),

    Dense(128, activation='relu'),
    BatchNormalization(),

    Dense(64, activation='relu'),
    BatchNormalization(),

    Dense(32, activation='relu'),
    BatchNormalization(),

    Dropout(0.3),

    Dense(10, activation='softmax')
])

deep_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

deep_model.summary()
