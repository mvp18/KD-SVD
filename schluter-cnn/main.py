import os
import numpy as np
import keras
import tensorflow as tf
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import h5py
import argparse

from load_data import *
from model import *
from config_cnn import *

# set gpu number 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()

parser.add_argument('-bs', '--batch_size', help="batch size used for training", default=32, type=int)
parser.add_argument('-lr', '--learning_rate', help="learning rate for sgd", default=1e-4, type=float)
parser.add_argument('-dr', '--drop_rate', default=0.2, type=float)
parser.add_argument('-e', '--num_epochs', help="number of epochs to run", default=50, type=int)
parser.add_argument('-es', '--early_stop', default=5, type=int)
parser.add_argument('-rd_lr', '--reduce_lr', default=3, type=int)
parser.add_argument('-seed', '--rand_seed', default=0, type=int)

args = parser.parse_args()

np.random.seed(args.rand_seed)

model = Schluter_CNN(args.drop_rate)

opt = SGD(lr=args.learning_rate, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print(model.summary())

score_string = 'val_acc-{val_accuracy:.4f}_tr_acc-{accuracy:.4f}_bestEp-{epoch:02d}'
model_save_name = './weights/'+score_string+'_bs-'+str(args.batch_size)+'_lr-'+str(args.learning_rate)+'_dr-'+str(args.drop_rate)+'.h5'

checkpoint = ModelCheckpoint(filepath=model_save_name, monitor='val_accuracy', verbose=1, save_weights_only=False, save_best_only=True, mode='auto')

earlyStopping = EarlyStopping(monitor='val_accuracy', patience=args.early_stop, verbose=1, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=args.reduce_lr, verbose=1, min_lr=1e-8)

X_tr, y_train = load_xy_data(None, MEL_JAMENDO_DIR, JAMENDO_LABEL_DIR, 'train')
Y_tr = np_utils.to_categorical(y_train, 2)

X_val, y_val = load_xy_data(None, MEL_JAMENDO_DIR, JAMENDO_LABEL_DIR, 'valid')
Y_val = np_utils.to_categorical(y_val, 2)

print("Train Data Shape", X_tr.shape, Y_tr.shape)
print("Val Data Shape", X_val.shape, Y_val.shape)

model.fit(X_tr, Y_tr, batch_size=args.batch_size, epochs=args.num_epochs, shuffle=True, validation_data=(X_val, Y_val), 
	       callbacks=[checkpoint, earlyStopping, reduce_lr])

print("Finished!")
