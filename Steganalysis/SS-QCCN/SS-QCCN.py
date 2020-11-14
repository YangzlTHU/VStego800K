#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Implementation of SS-QCCN algorithm
-------------
Based on paper:
    Steganalysis of QIM Steganography in Low-Bit-Rate Speech Signals
-------------

'''

import os, random, pickle, csv, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
from tqdm import tqdm

FOLD = 3  # = NUM_SAMPLE / number of testing samples
NUM_PCA_FEATURE = 300  # number of PCA features
NUM_SAMPLE = 5000  # total number of samples used for training
TEST_NUM_SAMPLE = 1200 # number of samples used for test

'''
SS-QCCN feature extraction
-------------
input
    file
        The path to an ASCII file.
        Each line contains three integers: x1 x2 x3, which are the three codewords of the frame.
        There are (number of frame) lines in total. 
output
    A numpy vector, which contains the features determined by SS-QCCN algorithm.
'''

def G729_SS_QCCCN(file):
    data = []
    with open(file, "r") as f:
        for line in f:
            line = [int(i) for i in line.split()]
            data.append(line)

    a = np.zeros(shape=(128, 128))
    c1 = np.zeros(shape=128)
    p = np.zeros(shape=(32, 32))
    c2 = np.zeros(shape=32)

    for i in range(len(data) - 1):
        data1 = data[i]
        data2 = data[i + 1]
        c1[data1[0]] += 1
        c2[data1[1]] += 1
        a[data1[0], data2[0]] += 1
        p[data1[1], data1[2]] += 1

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if c1[i] != 0:
                a[i, j] /= c1[i]

    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if c2[i] != 0:
                p[i, j] /= c2[i]

    return np.concatenate([a.reshape(128 * 128), p.reshape(32 * 32)])


'''
SS-QCCN training and testing
-------------
input
    positive_data_folder
        The folder that contains positive data files for training.
    negative_data_folder
        The folder that contains negative data files for training.
    t_positive_data_folder
        The folder that contains positive data files for testing.
    t_negative_data_folder
        The folder that contains negative data files for testing.
        The folder that contains negative data files for testing.
    result_folder
        The folder that stores the results.  
'''


def main(positive_data_folder, negative_data_folder, t_positive_data_folder, t_negative_data_folder, result_folder):
    build_model = G729_SS_QCCCN

    positive_data_files = [os.path.join(positive_data_folder, path) for path in os.listdir(positive_data_folder)]
    negative_data_files = [os.path.join(negative_data_folder, path) for path in os.listdir(negative_data_folder)]

    t_positive_data_files = [os.path.join(t_positive_data_folder, path) for path in
                             os.listdir(t_positive_data_folder)]
    t_negative_data_files = [os.path.join(t_negative_data_folder, path) for path in
                             os.listdir(t_negative_data_folder)]

    train_positive_data_files = positive_data_files[0:NUM_SAMPLE]  # The positive samples for training
    train_negative_data_files = negative_data_files[0:NUM_SAMPLE]  # The negative samples for training

    test_positive_data_files = t_positive_data_files[0:TEST_NUM_SAMPLE]  # The positive samples for testing
    test_negative_data_files = t_negative_data_files[0:TEST_NUM_SAMPLE]  # The negative samples for testing

    num_train_files = len(train_negative_data_files)
    num_test_files = len(test_negative_data_files)
    print(num_test_files)

    # calculate PCA matrix
    print("Calculating PCA matrix")

    feature = []

    for i in tqdm(range(num_train_files)):
        new_feature = build_model(train_negative_data_files[i])
        feature.append(new_feature)
    for i in tqdm(range(num_train_files)):
        new_feature = build_model(train_positive_data_files[i])
        feature.append(new_feature)
    feature = np.row_stack(feature)
    pca = PCA(n_components=NUM_PCA_FEATURE)
    pca.fit(feature)

    with open(os.path.join(result_folder, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)

    # load train data
    print("Loading train data")
    X = []
    Y = []
    for i in tqdm(range(num_train_files)):
        new_feature = build_model(train_negative_data_files[i])
        X.append(pca.transform(new_feature.reshape(1, -1)))
        Y.append(0)
    for i in tqdm(range(num_train_files)):
        new_feature = build_model(train_positive_data_files[i])
        X.append(pca.transform(new_feature.reshape(1, -1)))
        Y.append(1)
    X = np.row_stack(X)

    # train SVM
    print("Training SVM")
    clf = svm.SVC()
    clf.fit(X, Y)
    with open(os.path.join(result_folder, "svm.pkl"), "wb") as f:
        pickle.dump(clf, f)

    # test
    print("Testing")
    X = []
    Y = []
    for i in tqdm(range(num_test_files)):
        new_feature = build_model(test_negative_data_files[i])
        X.append(pca.transform(new_feature.reshape(1, -1)))
        Y.append(0)
    for i in tqdm(range(num_test_files)):
        new_feature = build_model(test_positive_data_files[i])
        X.append(pca.transform(new_feature.reshape(1, -1)))
        Y.append(1)
    X = np.row_stack(X)
    Y_predict = clf.predict(X)
    with open(os.path.join(result_folder, "Y_predict.pkl"), "wb") as f:
        pickle.dump(Y_predict, f)

    # output result

    true_negative = 0
    false_negative = 0
    true_positive = 0
    false_positive = 0

    print("Outputing result")
    with open(os.path.join(result_folder, "test_result.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "real class", "predict class"])
        for i in range(num_test_files):
            writer.writerow([test_negative_data_files[i], 0, Y_predict[i]])
            if Y_predict[i] == 0:
                true_negative += 1
            else:
                false_positive += 1

        for i in range(num_test_files):
            writer.writerow([test_positive_data_files[i], 1, Y_predict[i + num_test_files]])
            if Y_predict[i + num_test_files] == 1:
                true_positive += 1
            else:
                false_negative += 1

        writer.writerow(["num of test files", 2 * num_test_files])
        writer.writerow(["True Positive", true_positive])
        writer.writerow(["True Negative", true_negative])
        writer.writerow(["False Positive", false_positive])
        writer.writerow(["False Negative", false_negative])
        writer.writerow(["Accuracy", float(true_negative + true_positive) / (num_test_files * 2)])
        writer.writerow(["Precision", float(true_positive) / (true_positive + false_positive)])
        writer.writerow(["Recall", float(true_positive) / (true_positive + false_negative)])


if __name__ == "__main__":
    main('/data/data_QCCN/positive', '/data/data_QCCN/negative', '/data/data_QCCN/t_positive',
         '/data/data_QCCN/t_negative', '.')
