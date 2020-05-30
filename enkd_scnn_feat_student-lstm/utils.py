import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy as logloss
from tensorflow.keras.losses import kullback_leibler_divergence
from tensorflow.keras.metrics import categorical_accuracy
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import argparse
import h5py

from load_data import *
from model import *
from config_cnn import *

def acc(y_true, y_pred):
	y_true_hard = y_true[:, :2]
	y_pred_hard = y_pred[:, :2]
	return categorical_accuracy(y_true_hard, y_pred_hard)

def categorical_crossentropy(y_true, y_pred):
	y_true_hard = y_true[:, :2]
	y_pred_hard = y_pred[:, :2]
	return logloss(y_true_hard, y_pred_hard)

def kld_loss(y_true, y_pred):     
	y_true_softs = y_true[:, 2:]
	y_pred_softs = y_pred[:, 2:]
	return kullback_leibler_divergence(y_true_softs, y_pred_softs)

def kd_loss(alpha, temperature):

	def custom_loss(y_true, y_pred):

		y_true, y_true_softs = y_true[:, :2], y_true[:, 2:]
		y_pred, y_pred_softs = y_pred[:, :2], y_pred[:, 2:]
		
		loss = (1-alpha)*logloss(y_true, y_pred) + alpha*(temperature**2)*kullback_leibler_divergence(y_true_softs, y_pred_softs)
	
		return loss

	return custom_loss

def sample_scores(loaded_model, model_type, song):

	x_test, y_test = load_xy_data(song, MEL_JAMENDO_DIR, JAMENDO_LABEL_DIR)

	y_pred = loaded_model.predict(x_test, verbose=1)

	y_pred = y_pred[:, :2]

	y_pred = np.argmax(y_pred, axis=1)

	y_pred = y_pred.reshape(-1).astype(int)
	y_test = y_test.reshape(-1).astype(int)

	accuracy_single = (len(y_test) - np.sum(np.abs(y_pred - y_test)))/ len(y_test)
	f1 = f1_score(y_test, y_pred, average='binary')
	pr = precision_score(y_test, y_pred, average='binary')
	re = recall_score(y_test, y_pred, average='binary')
	
	print('\nSample Scores...\n')
	print('Accuracy: ' + str(accuracy_single))
	print('Precision: ', pr)
	print('Recall: ', re)
	print('F1-score: ', f1)

	del x_test

	return y_pred, y_test

def test(model_name, model_type, wts_dir, args):

	if model_type=='student':
		model = RNN_small(timesteps=CNN_INPUT_SIZE)
	elif model_type=='teacher':
		model = Leglaive_RNN(timesteps=CNN_INPUT_SIZE)	
	
	model.pop()
	model_logits = model.layers[-1].output
	model_logits_T = Lambda(lambda x: x/args.temperature)(model_logits)
	probs_T = Softmax(axis=1)(model_logits_T)
	probs_1 = Softmax(axis=1)(model_logits)
	output = Concatenate()([probs_1, probs_T])
	model = Model(inputs=model.input, outputs=output)

	model.load_weights(wts_dir+model_name)

	print(model.summary())

	list_of_songs = os.listdir(MEL_JAMENDO_DIR + 'test')

	y_preds = []
	y_tests = []

	for song in list_of_songs:
		
		y_pred, y_test = sample_scores(model, model_type, song)

		for i in range(len(y_pred)):
			y_preds.append(y_pred[i])
			y_tests.append(y_test[i])

	# convert list to np array 
	y_preds = np.array(y_preds)
	y_tests = np.array(y_tests)

	# calculate scores 
	acc = (len(y_tests) - np.sum(np.abs(y_preds - y_tests))) / float(len(y_tests))

	f1 = f1_score(y_tests, y_preds, average='binary')
	pr = precision_score(y_tests, y_preds, average='binary')
	re = recall_score(y_tests, y_preds, average='binary')

	tn, fp, fn, tp = confusion_matrix(y_tests, y_preds).ravel()
	fp_rate = fp / (fp + tn)
	fn_rate = fn / (fn + tp)

	print("TEST SCORES OVERALL\n")
	print('Acc %.4f' % acc)
	print('Precision %.4f' % pr)
	print('Recall %.4f' % re)
	print('F1-score %.4f' % f1)
	print('fp rate', fp_rate, 'fn_rate', fn_rate)

	scores = [acc, pr, re, f1, fp_rate, fn_rate]

	return scores