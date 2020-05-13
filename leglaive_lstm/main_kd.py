import os
import numpy as np
import pandas as pd
from scipy.special import softmax
import sys
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import h5py
import argparse

from load_data import *
from model import *
from config_rnn import *
from utils import *

# set gpu number 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()

parser.add_argument('-bs', '--batch_size', help="batch size used for training", default=64, type=int)
parser.add_argument('-lr', '--learning_rate', help="learning rate for adam", default=1e-4, type=float)
parser.add_argument('-temp', '--temperature', help="distillation temperature(>1)", default=2, type=float)
parser.add_argument('-alpha', '--alpha', help="weight to cce loss with hard targets, if None, wd be set to (1/T^2)", default=None, type=float)
parser.add_argument('-e', '--num_epochs', help="number of epochs to run", default=50, type=int)
parser.add_argument('-es', '--early_stop', default=7, type=int)
parser.add_argument('-rd_lr', '--reduce_lr', default=5, type=int)
parser.add_argument('-seed', '--rand_seed', default=0, type=int)

args = parser.parse_args()

np.random.seed(args.rand_seed)

teacher = tf.keras.models.load_model('./weights/'+'teacher_val_acc-0.8403_tr_acc-0.9234_bestEp-02_bs-64_lr-0.0001.h5')
student = RNN_small(timesteps=RNN_INPUT_SIZE)

print('Teacher Model:\n')
print(teacher.summary())

opt = Adam(lr=args.learning_rate)

# Removing softmax layers
teacher.pop()
student.pop()

student_logits = student.layers[-1].output
student_logits_T = Lambda(lambda x: x/args.temperature)(student_logits)
probs_T = Softmax(axis=2)(student_logits_T)
probs_1 = Softmax(axis=2)(student_logits)
output = Concatenate()([probs_1, probs_T])
student = Model(inputs=student.input, outputs=output)

if args.alpha==None: args.alpha=1/(args.temperature**2)

student.compile(optimizer=opt, loss=lambda y_true, y_pred: kd_loss(y_true, y_pred, args.alpha), 
                metrics=[acc, categorical_crossentropy, soft_logloss])

print('\nStudent Model:\n')
print(student.summary())

X_tr, y_train = load_xy_data(None, MEL_JAMENDO_DIR, JAMENDO_LABEL_DIR, 'train')
Y_tr = to_categorical(y_train, 2)

X_val, y_val = load_xy_data(None, MEL_JAMENDO_DIR, JAMENDO_LABEL_DIR, 'valid')
Y_val = to_categorical(y_val, 2)

print('\nLoading complete!\n')

teacher_tr_logits = teacher.predict(X_tr, verbose=1)
teacher_val_logits = teacher.predict(X_val, verbose=1)

Y_tr_soft = softmax(teacher_tr_logits/args.temperature, axis=2)
Y_tr = np.concatenate((Y_tr, Y_tr_soft), axis=2)

Y_val_soft = softmax(teacher_val_logits/args.temperature, axis=2)
Y_val = np.concatenate((Y_val, Y_val_soft), axis=2)

print("Train Data Shape", X_tr.shape, Y_tr.shape)
print("Val Data Shape", X_val.shape, Y_val.shape)

score_string = 'val_acc-{val_acc:.4f}_kd_tr_acc-{acc:.4f}_bestEp-{epoch:02d}'
model_save_name = './weights/'+score_string+'_bs-'+str(args.batch_size)+'_lr-'+str(args.learning_rate)+'_temp-'+str(args.temperature)+'_alpha-'+\
                  str(args.alpha)+'.h5'

checkpoint = ModelCheckpoint(filepath=model_save_name, monitor='val_acc', verbose=1, save_weights_only=False, save_best_only=True, mode='auto')
    
earlyStopping = EarlyStopping(monitor='val_acc', patience=args.early_stop, verbose=1, mode='auto')
    
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.8, patience=args.reduce_lr, verbose=1, min_lr=1e-8)

history = student.fit(X_tr, Y_tr, batch_size=args.batch_size, epochs=args.num_epochs, shuffle=True, validation_data=(X_val, Y_val), 
                    callbacks=[checkpoint, earlyStopping, reduce_lr])

tr_loss = history.history['loss']
val_loss = history.history['val_loss']
tr_acc = history.history['acc']
val_acc = history.history['val_acc']

idx, best_val_acc = np.argmax(val_acc), np.max(val_acc)
corr_tr_acc = tr_acc[idx]

df_save = pd.DataFrame({'tr_loss':tr_loss, 'val_loss':val_loss, 'tr_acc':tr_acc, 'val_acc':val_acc})

best_model_name = 'val_acc-'+'{0:.4f}_kd'.format(best_val_acc)+'_tr_acc-'+'{0:.4f}'.format(corr_tr_acc)+'_bestEp-'+'{:02d}'.format(idx+1)+\
                  '_bs-'+str(args.batch_size)+'_lr-'+str(args.learning_rate)+'_temp-'+str(args.temperature)+'_alpha-'+ str(args.alpha)+'.h5'

save_path = './results/'
if not os.path.exists(save_path): os.makedirs(save_path)

df_save = test(best_model_name, 'kd', df_save)
suffix = best_model_name[:-3]+'test_acc-'+'{0:.4f}'.format(df_save['test_acc'])
df_save.to_csv(open(save_path + suffix + '.csv', 'w'))

print("Finished!")