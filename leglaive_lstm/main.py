import os
import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import h5py
import argparse

from load_data import *
from model import *
from config_rnn import *

# set gpu number 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()

parser.add_argument('-model', '--model_type', help="teacher / student / kd (knowledge distillation)", default='teacher', type=str)
parser.add_argument('-bs', '--batch_size', help="batch size used for training", default=64, type=int)
parser.add_argument('-lr', '--learning_rate', help="learning rate for adam", default=1e-4, type=float)
parser.add_argument('-e', '--num_epochs', help="number of epochs to run", default=50, type=int)
parser.add_argument('-es', '--early_stop', default=7, type=int)
parser.add_argument('-rd_lr', '--reduce_lr', default=5, type=int)
parser.add_argument('-seed', '--rand_seed', default=0, type=int)

args = parser.parse_args()

np.random.seed(args.rand_seed)

if args.model_type=='teacher': model = Leglaive_RNN(timesteps=RNN_INPUT_SIZE)
elif args.model_type=='student': model = RNN_small(timesteps=RNN_INPUT_SIZE)
elif args.model_type=='kd':
    teacher = Leglaive_RNN(timesteps=RNN_INPUT_SIZE)
    student = RNN_small(timesteps=RNN_INPUT_SIZE)
else:
    print('Invalid model type specified!')
    sys.exit()

opt = Adam(lr=args.learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print(model.summary())

score_string = args.model_type+'_val_acc-{val_accuracy:.4f}_tr_acc-{accuracy:.4f}_bestEp-{epoch:02d}'
model_save_name = './weights/'+score_string+'_bs-'+str(args.batch_size)+'_lr-'+str(args.learning_rate)+'.h5'

checkpoint = ModelCheckpoint(filepath=model_save_name, monitor='val_accuracy', verbose=1, save_weights_only=False, save_best_only=True, mode='auto')
    
earlyStopping = EarlyStopping(monitor='val_accuracy', patience=args.early_stop, verbose=1, mode='auto')
    
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=args.reduce_lr, verbose=1, min_lr=1e-8)

X_tr, y_train = load_xy_data(None, MEL_JAMENDO_DIR, JAMENDO_LABEL_DIR, 'train')
Y_tr = to_categorical(y_train, 2)

X_val, y_val = load_xy_data(None, MEL_JAMENDO_DIR, JAMENDO_LABEL_DIR, 'valid')
Y_val = to_categorical(y_val, 2)

print("Train Data Shape", X_tr.shape, Y_tr.shape)
print("Val Data Shape", X_val.shape, Y_val.shape)

model.fit(X_tr, Y_tr, batch_size=args.batch_size, epochs=args.num_epochs, shuffle=True, validation_data=(X_val, Y_val), 
          callbacks=[checkpoint, earlyStopping, reduce_lr])

print("Finished training")