from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras import Sequential

deep_model = Sequential([
    Input(shape=(X_train.shape[1],)),

    Dense(64, activation='relu'),
    BatchNormalization(),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    
    Dense(16, activation='relu'),
    BatchNormalization(),

    Dense(16, activation='relu'),
    BatchNormalization(),

    Dense(8, activation='relu'),
    BatchNormalization(),

    Dropout(0.2),  # Reduces overfitting

    Dense(1, activation='sigmoid')   # Output layer
])

deep_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

deep_model.summary()
