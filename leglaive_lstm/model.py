from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, TimeDistributed

def Leglaive_RNN(timesteps, data_dim=80):
    # input shape : (batch_size, timestep, data_dim=80)
    model = Sequential()
    model.add(Bidirectional(LSTM(30, return_sequences=True), input_shape=(timesteps, data_dim)))
    model.add(Bidirectional(LSTM(20, return_sequences=True)))
    model.add(Bidirectional(LSTM(40, return_sequences=True)))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    return model
