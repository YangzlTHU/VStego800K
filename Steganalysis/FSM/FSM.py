#!/usr/bin/env python
# -*-coding:utf-8 -*-

'''
Implementation of Fast Steganalysis Method for VoIP Streams
-------------
Based on paper:
    Fast Steganalysis Method for VoIP Streams
-------------
'''
import numpy as np
import os, pickle, random, datetime
import os, random, pickle, csv, sys
from keras.layers import Reshape, Dense, Dropout, LSTM, Bidirectional, AveragePooling1D, Flatten, Input, MaxPooling1D, \
    Convolution1D, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Activation
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from sklearn.svm import SVC
from keras.layers import Embedding, Dense, Flatten, Input
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras import regularizers
import pdb
from keras import regularizers
import tensorflow as tf
import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import sklearn as sk

# from test_metric import *

Embedding_rate = 10
SAMPLE_LENGTH = 1000  # The sample length (ms)
BATCH_SIZE = 256  # batch size
# ITER = 200  # number of iteration
ITER = 20
FOLD = 5  # = NUM_SAMPLE / number of testing samples
NUM_FILTERS = 64
FILTER_SIZES = [5, 4, 3]
num_class = 2
vocab_size = 256
hidden_num = 64
k = 2
tag = 'lstm+cnn_iter'
flag_str = 'FM'

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

File_Embed = "/data/train/g729a_Steg_feat"
File_NoEmbed = "/data/train/g729a_0_feat"

FOLDERS = [
    {"class": 1, "folder": File_Embed},  # The folder that contains positive data files.
    {"class": 0, "folder": File_NoEmbed}  # The folder that contains negative data files.
]
''''
Get the paths of all files in the folder
'''


def get_file_list(folder):
    file_list = []
    for file in os.listdir(folder):
        file_list.append(os.path.join(folder, file))
    return file_list


'''
Read codeword file
-------------
input
    file_path
        The path to an ASCII file.
        Each line contains features.
        There are (number of frame) lines in total.
output
    the list of codewords
'''


def parse_sample(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    sample = []
    for line in lines:
        # line_split = line.strip("\r\n\t").strip().split("\t")
        line = [int(l) for l in line.split()]
        sample.append(line)
    return sample


'''
Save variable in pickle
'''


def save_variable(file_name, variable):
    file_object = open(file_name, "wb")
    pickle.dump(variable, file_object)
    file_object.close()


'''
teacher_model
'''


def teacher_model():
    model_input = Input(shape=(int(SAMPLE_LENGTH / 10), 5))
    # x_in = Flatten()(model_input)
    # print("test shape:",x_in.shape)
    max_size = 256
    hs = 64
    nb_head = 8
    nb_hs = 24
    emb_x = Embedding(256, 64)(model_input)
    x = Flatten()(emb_x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x_s = Dense(num_class)(x)
    outputs = Activation('softmax')(x_s)
    model = Model(inputs=model_input, outputs=outputs)
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    model.summary()

    return model


'''
student_model
'''


def student_model():
    model_input = Input(shape=(int(SAMPLE_LENGTH / 10), 5))
    # x_in = Flatten()(model_input)
    # print("test shape:",x_in.shape)
    max_size = 256
    hs = 64
    nb_head = 8
    nb_hs = 24
    emb_x = Embedding(256, 64)(model_input)
    O_seq = Flatten()(emb_x)
    O_seq = Dropout(0.5)(O_seq)
    x_s = Dense(num_class)(O_seq)
    outputs = Activation('softmax')(x_s)
    model = Model(inputs=model_input, outputs=outputs)
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    model.summary()
    return model


def softmax_with_temp(x):
    Temp = 1.0
    e_x = np.exp((x - x.max(axis=1, keepdims=True)) / Temp)
    out = e_x / e_x.sum(axis=1, keepdims=True)
    return out


'''
training and testing
'''
if __name__ == '__main__':
    all_files = [(item, folder["class"]) for folder in FOLDERS for item in get_file_list(folder["folder"])]
    random.shuffle(all_files)
    # save_variable('all_files.pkl', all_files)
    all_samples_x = [(parse_sample(item[0])) for item in all_files]
    all_samples_y = [item[1] for item in all_files]
    print(len(all_samples_y))

    np_all_samples_x = np.asarray(all_samples_x)
    np_all_samples_y = np.asarray(all_samples_y)
    print("np_all_samples_x ")
    print(np_all_samples_x.shape)
    print("np_all_samples_y ")
    print(np_all_samples_y.shape)

    # save_variable('np_all_samples_x.pkl', np_all_samples_x)
    # save_variable('np_all_samples_y.pkl', np_all_samples_y)
    file_num = len(np_all_samples_y)
    sub_file_num = int(file_num / FOLD)
    dataSize = np_all_samples_x.shape

    x_test = np_all_samples_x[0: sub_file_num]  # The samples for testing
    y_test_ori = np_all_samples_y[0: sub_file_num]  # The label of the samples for testing

    x_train = np_all_samples_x[sub_file_num:]  # The samples for training
    y_train_ori = np_all_samples_y[sub_file_num:]  # The label of the samples for training

    y_train = np_utils.to_categorical(y_train_ori, num_classes=2)
    y_test = np_utils.to_categorical(y_test_ori, num_classes=2)

    t_model = teacher_model()
    iter1 = 5
    iter2 = 3
    iter3 = 3
    print("-------------- train  teacher model--------------------------------")
    model_path = 't_%s_%d_%d.h5' % (flag_str, Embedding_rate, SAMPLE_LENGTH)
    if os.path.exists(model_path):
        model = t_model.load_weights(model_path)
        print("t load...")

    for i in range(iter1):
        print("teacher_model iter:", str(i))
        t_model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1, validation_data=(x_test, y_test))
        t_model.save_weights(model_path)
        # y_pred = t_model.predict(x_test)
        # y_pred = np.argmax(y_pred, axis=1)
        # print_result(y_test_ori, y_pred)

    print("---------------trian student model-------------------------------------------------")
    # recode 2
    s_model = student_model()
    model_path = 's_%s_%d_%d.h5' % (flag_str, Embedding_rate, SAMPLE_LENGTH)
    if os.path.exists(model_path):
        model = s_model.load_weights(model_path)

    for i in range(iter2):
        print("student_model iter:", str(i))
        s_model.fit(x_train, y_train, batch_size=64, epochs=1, validation_data=(x_test, y_test))
        s_model.save_weights('s_%s_%d_%d.h5' % (flag_str, Embedding_rate, SAMPLE_LENGTH))
        # y_pred = s_model.predict(x_test)
        # y_pred = np.argmax(y_pred, axis=1)
        # print_result(y_test_ori, y_pred)

    # get soft label
    t_out = t_model.predict(x_train)
    print(t_out.shape)
    # print(t_out.shape)
    for l in s_model.layers:
        l.trainable = True

    print("---------------knowledge distillation-------------------------------------------------")
    s_model_new = student_model()
    for i in range(iter3):
        print("final_model iter:", str(i))
        s_model_new.fit(x_train, t_out, batch_size=BATCH_SIZE, epochs=1, validation_data=(x_test, y_test))
        s_model_new.save('finalmodel_%s_%d_%d.h5' % (flag_str, Embedding_rate, SAMPLE_LENGTH))
        s_model_new.save_weights('finalweight_%s_%d_%d.h5' % (flag_str, Embedding_rate, SAMPLE_LENGTH))
        y_pred = s_model_new.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        # save_variable('y_pred', y_pred)
        # print_result(y_test_ori, y_pred)
    print("done")
