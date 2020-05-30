from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, Softmax, Concatenate, Lambda

def Leglaive_RNN(timesteps, data_dim=80):
    # input shape : (batch_size, timestep, data_dim=80)
    model = Sequential()
    model.add(Bidirectional(LSTM(30, return_sequences=True), input_shape=(timesteps, data_dim)))
    model.add(Bidirectional(LSTM(20, return_sequences=True)))
    model.add(Bidirectional(LSTM(40, return_sequences=False)))
    model.add(Dense(2))
    model.add(Softmax(axis=1))

    return model

def RNN_small(timesteps, data_dim=80):
    # input shape : (batch_size, timestep, data_dim=80)
    model = Sequential()
    model.add(Bidirectional(LSTM(30, return_sequences=False), input_shape=(timesteps, data_dim)))
    model.add(Dense(2))
    model.add(Softmax(axis=1))

    return model

if __name__ == '__main__':
    model = Leglaive_RNN(115, 80)
    print(model.summary())

