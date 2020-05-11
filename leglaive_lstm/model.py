from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, Softmax

def Leglaive_RNN(timesteps, data_dim=80):
    # input shape : (batch_size, timestep, data_dim=80)
    model = Sequential()
    model.add(Bidirectional(LSTM(30, return_sequences=True), input_shape=(timesteps, data_dim)))
    model.add(Bidirectional(LSTM(20, return_sequences=True)))
    model.add(Bidirectional(LSTM(40, return_sequences=True)))
    model.add(TimeDistributed(Dense(2)))
    model.add(Softmax(axis=2))

    return model

def RNN_small(timesteps, data_dim=80):
    # input shape : (batch_size, timestep, data_dim=80)
    model = Sequential()
    model.add(Bidirectional(LSTM(30, return_sequences=True), input_shape=(timesteps, data_dim)))
    model.add(TimeDistributed(Dense(2)))
    model.add(Softmax(axis=2))

    return model

if __name__ == '__main__':
    model = RNN_small(218)
    print(model.summary())

