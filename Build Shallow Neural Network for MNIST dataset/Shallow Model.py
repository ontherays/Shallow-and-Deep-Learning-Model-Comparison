from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

shallow_model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    BatchNormalization(),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    
    Dense(10, activation='softmax')
])

shallow_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

shallow_model.summary()
