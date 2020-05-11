import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import argparse
import h5py
from load_data import *
from config_rnn import *
# set gpu number 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('-model', '--model_name', default='val_acc-0.8403_tr_acc-0.9234_bestEp-02_bs-64_lr-0.0001.h5', type=str)
parser.add_argument('-data', '--dataset', default='jamendo', type=str)
parser.add_argument('-seed', '--rand_seed', default=0, type=int)

args = parser.parse_args()

np.random.seed(args.rand_seed)

def test(model_name, test_set, song=None):
    ''' 
    Test the model
    '''
    if test_set == 'jamendo':
        x_test, y_test = load_xy_data(song, MEL_JAMENDO_DIR, JAMENDO_LABEL_DIR)
    elif test_set == 'vibrato':
        x_test, y_test = load_xy_data(song, MEL_VIB_DIR, '', model_name)
    elif test_set[:3] == 'voc':
        MLD = '../loudness/' + test_set + '/leglaive_mel_dir/'
        x_test, y_test = load_xy_data_mdb(song, MLD, MDB_LABEL_DIR, model_name)

    y_pred = np.argmax(loaded_model.predict(x_test, verbose=1), axis=2)

    y_pred = y_pred.reshape(-1).astype(int)
    y_test = y_test.reshape(-1).astype(int)

    accuracy_single = (len(y_test) - np.sum(np.abs(y_pred - y_test))) * 100.0 / len(y_test)
    f1 = f1_score(y_test, y_pred, average='binary')
    pr = precision_score(y_test, y_pred, average='binary')
    re = recall_score(y_test, y_pred, average='binary')
    
    print('Sample Scores...\n')
    print('Accuracy: ' + str(accuracy_single))
    print('Precision: ', pr)
    print('Recall: ', re)
    print('F1-score: ', f1)

    return y_pred, y_test


if __name__ == '__main__':
    # best : weights/rnn_20180531-1 
    if args.dataset == 'jamendo':
        test_sets = ['jamendo']
    elif args.dataset == 'vibrato':
        test_sets = ['vibrato']
    elif args.dataset == 'snr':
        test_sets = ['voc_p0', 'voc_p6', 'voc_p12', 'voc_m6', 'voc_m12']
    else:
        print("unknown dataset")
        sys.exit()

    # load model
    model_name = args.model_name
    loaded_model = tf.keras.models.load_model('./weights/'+model_name)
    print(loaded_model.summary())

    for test_set in test_sets:
        predicted_values = {}  # for saving the frame level prediction
        y_preds = []
        y_tests = []

        if test_set == 'jamendo':
            list_of_songs = os.listdir(MEL_JAMENDO_DIR + 'test')
        elif test_set == 'vibrato':  # vibrato test
            list_of_songs = os.listdir(MEL_VIB_DIR)
        elif test_set[:3] == 'voc':  # snr test
            MLD = '../loudness/' + test_set + '/leglaive_mel_dir/'
            list_of_songs = os.listdir(MLD)

        for song in list_of_songs:
            y_pred, y_test = test(model_name, test_set, song)

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
