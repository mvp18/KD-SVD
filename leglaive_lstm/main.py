import os
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.activations import softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import h5py
import argparse

from load_data import *
from model import *
from config_rnn import *
from utils import test

# set gpu number 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()

parser.add_argument('-model', '--model_type', help="teacher/student", default='teacher', type=str)
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
else:
    print('Invalid model type specified!')
    sys.exit()

opt = Adam(lr=args.learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
print(model.summary())

score_string = 'val_acc-{val_acc:.4f}_'+args.model_type+'_tr_acc-{acc:.4f}_bestEp-{epoch:02d}'
model_save_name = './weights/'+score_string+'_bs-'+str(args.batch_size)+'_lr-'+str(args.learning_rate)+'.h5'

checkpoint = ModelCheckpoint(filepath=model_save_name, monitor='val_acc', verbose=1, save_weights_only=False, save_best_only=True, mode='auto')
    
earlyStopping = EarlyStopping(monitor='val_acc', patience=args.early_stop, verbose=1, mode='auto')
    
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.8, patience=args.reduce_lr, verbose=1, min_lr=1e-8)

X_tr, y_train = load_xy_data(None, MEL_JAMENDO_DIR, JAMENDO_LABEL_DIR, 'train')
Y_tr = to_categorical(y_train, 2)

X_val, y_val = load_xy_data(None, MEL_JAMENDO_DIR, JAMENDO_LABEL_DIR, 'valid')
Y_val = to_categorical(y_val, 2)

print("Train Data Shape", X_tr.shape, Y_tr.shape)
print("Val Data Shape", X_val.shape, Y_val.shape)

history = model.fit(X_tr, Y_tr, batch_size=args.batch_size, epochs=args.num_epochs, shuffle=True, validation_data=(X_val, Y_val), 
                    callbacks=[checkpoint, earlyStopping, reduce_lr])

tr_loss = history.history['loss']
val_loss = history.history['val_loss']
tr_acc = history.history['acc']
val_acc = history.history['val_acc']

idx, best_val_acc = np.argmax(val_acc), np.max(val_acc)
corr_tr_acc = tr_acc[idx]

df_save = pd.DataFrame({'tr_loss':tr_loss, 'val_loss':val_loss, 'tr_acc':tr_acc, 'val_acc':val_acc})

best_model_name = 'val_acc-'+'{0:.4f}_'.format(best_val_acc)+args.model_type+'_tr_acc-'+'{0:.4f}'.format(corr_tr_acc)+\
				  '_bestEp-'+'{:02d}'.format(idx+1)+'_bs-'+str(args.batch_size)+'_lr-'+str(args.learning_rate)+'.h5'

save_path = './results/'
if not os.path.exists(save_path): os.makedirs(save_path)

df_save = test(best_model_name, args.model_type, df_save, args)
suffix = best_model_name[:-3]+'test_acc-'+'{0:.4f}_'.format(df_save['test_acc'])
df_save.to_csv(open(save_path + suffix + '.csv', 'w'))

print("Finished!")