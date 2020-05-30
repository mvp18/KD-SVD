import os
import shutil
import numpy as np
import pandas as pd
import time
from scipy.special import softmax
import sys
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import h5py
import argparse

from load_data import *
from model import *
from config_cnn import *
from utils import *

# set gpu number 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()

parser.add_argument('-fs', '--filter_scale', help="1 for teacher, >1 for smaller student models (power of 2 and <=32)", default=2, type=int)
parser.add_argument('-bs', '--batch_size', help="batch size used for training", default=32, type=int)
parser.add_argument('-lr', '--learning_rate', help="learning rate for sgd", default=1e-4, type=float)
parser.add_argument('-dr', '--drop_rate', default=0.2, type=float)
parser.add_argument('-temp', '--temperature', help="distillation temperature(>1)", default=5, type=float)
parser.add_argument('-alpha', '--alpha', help="weight to cce loss with soft targets, 1-alpha to loss with hard targets", default=0.2, type=float)
parser.add_argument('-comb', '--combination', help="combining predictions of models in ensemble (am/gm)", default='am', type=str)
parser.add_argument('-e', '--num_epochs', help="number of epochs to run", default=50, type=int)
parser.add_argument('-es', '--early_stop', default=5, type=int)
parser.add_argument('-rd_lr', '--reduce_lr', default=3, type=int)
parser.add_argument('-seed', '--rand_seed', default=0, type=int)

args = parser.parse_args()

np.random.seed(args.rand_seed)

teacher_lstm = tf.keras.models.load_model('../lstm_scnn_feat/val_acc-0.6886_teacher_tr_acc-0.9730_bestEp-09_bs-64_lr-0.0001.h5')
teacher_cnn = tf.keras.models.load_model('../schluter-cnn/teacher_val_acc-0.7920_tr_acc-0.9405_bestEp-05_bs-32_lr-0.0001_dr-0.2_fs-1.h5')

student = Schluter_CNN(args.drop_rate, args.filter_scale)

opt = SGD(lr=args.learning_rate, momentum=0.9, nesterov=True)

# Removing softmax layers
teacher_lstm.pop()
teacher_cnn.pop()
student.pop()

student_logits = student.layers[-1].output
student_logits_T = Lambda(lambda x: x/args.temperature)(student_logits)
probs_T = Softmax(axis=1)(student_logits_T)
probs_1 = Softmax(axis=1)(student_logits)
output = Concatenate()([probs_1, probs_T])
student = Model(inputs=student.input, outputs=output)

student.compile(optimizer=opt, loss=kd_loss(args.alpha, args.temperature), metrics=[acc, categorical_crossentropy, kld_loss])

print('\nStudent Model:\n')
print(student.summary())

X_tr, y_train = load_xy_data(None, MEL_JAMENDO_DIR, JAMENDO_LABEL_DIR, 'train')
Y_tr = to_categorical(y_train, 2)

X_val, y_val = load_xy_data(None, MEL_JAMENDO_DIR, JAMENDO_LABEL_DIR, 'valid')
Y_val = to_categorical(y_val, 2)

# Train Logits
teacher_lstm_tr_logits = teacher_lstm.predict(np.swapaxes(np.squeeze(X_tr, axis=3), 1, 2), verbose=1)
teacher_cnn_tr_logits = teacher_cnn.predict(X_tr, verbose=1)

teacher_lstm_tr_prob_T = softmax(teacher_lstm_tr_logits/args.temperature, axis=1)
teacher_cnn_tr_prob_T = softmax(teacher_cnn_tr_logits/args.temperature, axis=1)

# Val Logits
teacher_lstm_val_logits = teacher_lstm.predict(np.swapaxes(np.squeeze(X_val, axis=3), 1, 2), verbose=1)
teacher_cnn_val_logits = teacher_cnn.predict(X_val, verbose=1)

teacher_lstm_val_prob_T = softmax(teacher_lstm_val_logits/args.temperature, axis=1)
teacher_cnn_val_prob_T = softmax(teacher_cnn_val_logits/args.temperature, axis=1)

if args.combination=='am': 
	Y_tr_soft = (teacher_lstm_tr_prob_T+teacher_cnn_tr_prob_T)/2
	Y_val_soft = (teacher_lstm_val_prob_T+teacher_cnn_val_prob_T)/2
elif args.combination=='gm':
	GM = np.sqrt(teacher_lstm_tr_prob_T*teacher_cnn_tr_prob_T)
	GM /= GM.sum(axis=1)[:, np.newaxis]
	Y_tr_soft = GM
	GM = np.sqrt(teacher_lstm_val_prob_T*teacher_cnn_val_prob_T)
	GM /= GM.sum(axis=1)[:, np.newaxis]
	Y_val_soft = GM

Y_tr = np.concatenate((Y_tr, Y_tr_soft), axis=1)
Y_val = np.concatenate((Y_val, Y_val_soft), axis=1)

print('\nLoading complete!\n')

print("Train Data Shape", X_tr.shape, Y_tr.shape)
print("Val Data Shape", X_val.shape, Y_val.shape)

timestampTime = time.strftime("%H%M%S")
timestampDate = time.strftime("%d%m%Y")
timestampLaunch = timestampDate + '_' + timestampTime

wts_dir = './weights_'+timestampLaunch+'/'
if not os.path.exists(wts_dir): os.makedirs(wts_dir)

score_string = 'val_acc-{val_acc:.4f}_tr_acc-{acc:.4f}_bestEp-{epoch:02d}'
model_save_name = wts_dir+score_string+'_bs-'+str(args.batch_size)+'_lr-'+str(args.learning_rate)+'_dr-'+str(args.drop_rate)+\
				  '_temp-'+str(args.temperature)+'_alpha-'+str(args.alpha)+'.h5'

checkpoint = ModelCheckpoint(filepath=model_save_name, monitor='val_acc', verbose=1, save_weights_only=True, save_best_only=True, mode='auto')
    
earlyStopping = EarlyStopping(monitor='val_acc', patience=args.early_stop, verbose=1, mode='auto')
    
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=args.reduce_lr, verbose=1, min_lr=1e-8)

history = student.fit(X_tr, Y_tr, batch_size=args.batch_size, epochs=args.num_epochs, shuffle=True, validation_data=(X_val, Y_val), 
                      callbacks=[checkpoint, earlyStopping, reduce_lr])

tr_loss = history.history['loss']
val_loss = history.history['val_loss']
tr_acc = history.history['acc']
val_acc = history.history['val_acc']

idx, best_val_acc = np.argmax(val_acc), np.max(val_acc)
corr_tr_acc = tr_acc[idx]

df_save = pd.DataFrame({'tr_loss':tr_loss, 'val_loss':val_loss, 'tr_acc':tr_acc, 'val_acc':val_acc})

best_model_name = 'val_acc-'+'{0:.4f}'.format(best_val_acc)+'_tr_acc-'+'{0:.4f}'.format(corr_tr_acc)+'_bestEp-'+'{:02d}'.format(idx+1)+\
                  '_bs-'+str(args.batch_size)+'_lr-'+str(args.learning_rate)+'_dr-'+str(args.drop_rate)+'_temp-'+str(args.temperature)+\
                  '_alpha-'+ str(args.alpha)+'.h5'

save_path = './results/'+'fs'+str(args.filter_scale)+'/'+args.combination+'/'
if not os.path.exists(save_path): os.makedirs(save_path)

scores = test(best_model_name, wts_dir, args)
suffix = best_model_name[:-3]+'_acc-{0:.4f}'.format(scores[0])+'_pr-{0:.4f}'.format(scores[1])+'_re-{0:.4f}'.format(scores[2])+\
        '_f1-{0:.4f}'.format(scores[3])+'_fp-{0:.4f}'.format(scores[4])+'_fn-{0:.4f}'.format(scores[5])

df_save.to_csv(open(save_path + suffix + '.csv', 'w'))

if os.path.exists(wts_dir): shutil.rmtree(wts_dir)

print("Finished!")