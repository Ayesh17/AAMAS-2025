# lstm_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking

def create_lstm_model(timesteps, num_features, num_classes):
    model = Sequential()
    # Add a masking layer to ignore zero values in the input
    model.add(Masking(mask_value=0.0, input_shape=(timesteps, num_features)))
    # LSTM layers
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
