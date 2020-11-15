#!/usr/bin/env python
# -*-coding:utf-8 -*-
from __future__ import division
import numpy as np
import os, pickle, random, datetime
import os, random, pickle, csv, sys
import time
import argparse
import logging
from collections import OrderedDict
import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
import random

# 将模型load 然后使用测试集进行测试
test_File_Embed = "/data/data_SFFN/g729a_Steg_test_feat"  # "/data1/linzn/data/ch_g729a_%d_%dms_FEAT" % (Embedding_rate, SAMPLE_LENGTH)
test_File_NoEmbed = "/data/data_SFFN/g729a_0_test_feat"
test_FOLDERS = [
    {"class": 1, "folder": test_File_Embed},  # The folder that contains positive data files.
    {"class": 0, "folder": test_File_NoEmbed}  # The folder that contains negative data files.
]
SAMPLE_LENGTH = 1000
num_class = 2


def get_file_list(folder):
    file_list = []
    for file in os.listdir(folder):
        file_list.append(os.path.join(folder, file))
    return file_list


def parse_sample(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    sample = []
    for line in lines:
        # line_split = line.strip("\r\n\t").strip().split("\t")
        line = [int(l) for l in line.split()]
        sample.append(line)
    return sample


class CLS(nn.Module):
    def __init__(self, in_size, n_class):
        super(CLS, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(in_size, n_class), nn.Softmax(dim=1))

    def forward(self, x):
        out = self.classifier(x)
        return out


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.mp = nn.MaxPool2d(kernel_size=2)
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.mp(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mp(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.ap(x)
        x = torch.squeeze(x)
        return x


class MODEL(nn.Module):
    def __init__(self, opt):
        super(MODEL, self).__init__()
        self.rnn1 = nn.LSTM(input_size=opt.input_dim1, num_layers=opt.num_layers, hidden_size=opt.hidden_size,
                            batch_first=True)
        self.rnn2 = nn.LSTM(input_size=opt.input_dim2, num_layers=opt.num_layers, hidden_size=opt.hidden_size,
                            batch_first=True)
        self.cnn = CNN()
        self.cls = CLS(in_size=128, n_class=opt.n_class)
        self.cls1 = CLS(in_size=opt.hidden_size, n_class=opt.n_class)
        self.cls2 = CLS(in_size=opt.hidden_size, n_class=opt.n_class)

        if torch.cuda.is_available():
            self.rnn1.cuda()
            self.rnn2.cuda()
            self.cnn.cuda()
            self.cls.cuda()
            self.cls1.cuda()
            self.cls2.cuda()

        self.params = list(self.cnn.parameters()) + list(self.cls.parameters()) + list(self.rnn1.parameters()) + list(
            self.cls1.parameters()) + list(self.rnn2.parameters()) + list(self.cls2.parameters())

    def load_state_dict(self, state_dict):
        self.rnn1.load_state_dict(state_dict[0])
        self.cls1.load_state_dict(state_dict[1])
        self.rnn2.load_state_dict(state_dict[2])
        self.cls2.load_state_dict(state_dict[3])
        self.cnn.load_state_dict(state_dict[4])
        self.cls.load_state_dict(state_dict[5])

    def state_dict(self):
        state_dict = [self.rnn1.state_dict(), self.cls1.state_dict(), self.rnn2.state_dict(), self.cls2.state_dict(),
                      self.cnn.state_dict(), self.cls.state_dict()]
        return state_dict

    def val_start(self):
        """switch to evaluate mode
        """
        self.rnn1.eval()
        self.cls1.eval()
        self.rnn2.eval()
        self.cls2.eval()
        self.cnn.eval()
        self.cls.eval()

        for param in self.params:
            param.requires_grad = False

    def forward_pred(self, x):
        x1 = x[:, :, 0:opt.input_dim1]
        x2 = x[:, :, opt.input_dim1:opt.input_dim1 + opt.input_dim2]
        embed1, _ = self.rnn1(x1)
        embed2, _ = self.rnn2(x2)
        embed_cat = torch.stack((embed1, embed2), 1)
        embed = self.cnn(embed_cat)
        pred = self.cls(embed)
        return pred


def get_loaders(opt):
    # experiment_data_path = os.path.join(opt.root_path, "experiment_data")

    # load training data
    '''
    train_positive = np.load(os.path.join(experiment_data_path, "%s_%s_%s_%sms_LSFCW_train.npy" % (
        opt.language, opt.code, opt.algorithm, opt.time))).astype('float32')

    train_negative = np.load(
        os.path.join(experiment_data_path, "%s_%s_0_%sms_LSFCW_train.npy" % (opt.language, opt.code, opt.time))).astype(
        'float32')
    '''
    test_positive_files = [(item, folder["class"]) for folder in FOLDERS1 for item in get_file_list(folder["folder"])]
    test_positive = [(parse_sample(item[0])) for item in train_positive_files]

    test_negative_files = [(item, folder["class"]) for folder in FOLDERS0 for item in get_file_list(folder["folder"])]
    test_negative = [(parse_sample(item[0])) for item in train_negative_files]
    num_val = int(len(train_negative_files) / 5)
    opt.num_val = num_val

    test_positive = np.array(test_positive).astype('float32')
    test_negative = np.array(test_negative).astype('float32')

    opt.num_test = num_test
    print('num_test', num_test)

    # test tensor
    test_negative_tersor = torch.tensor(test_negative)
    test_positive_tersor = torch.tensor(test_positive)
    y_test_negative_tersor = torch.tensor(np.zeros(opt.num_test, ), dtype=torch.int64)
    y_test_positive_tersor = torch.tensor(np.ones(opt.num_test, ), dtype=torch.int64)

    # TensorDataset

    val_negative_dataset = Data.TensorDataset(val_negative_tersor, y_val_negative_tersor)
    val_positive_dataset = Data.TensorDataset(val_positive_tersor, y_val_positive_tersor)

    # DataLoader
    test_negative_loader = Data.DataLoader(dataset=test_negative_dataset, batch_size=opt.batch_size)
    test_positive_loader = Data.DataLoader(dataset=test_positive_dataset, batch_size=opt.batch_size)

    return test_negative_loader, test_positive_loader


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def train(opt, train_loader, model, epoch):
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    for i, train_data in enumerate(train_loader):
        # make sure train logger is used
        model.logger = train_logger
        # Update the model
        # print(type(*train_data))
        model.train_emb(*train_data)
        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                    .format(
                    epoch, i, len(train_loader), e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)


def test_validate(opt, all_files, model):
    # compute the predictions
    true_negative = 0
    false_negative = 0
    true_positive = 0
    false_positive = 0
    all_samples_x = [(parse_sample(item[0])) for item in all_files]
    all_samples_y = [item[1] for item in all_files]
    np_all_samples_x = np.asarray(all_samples_x).astype('float32')
    np_all_samples_y = np.asarray(all_samples_y).astype('float32')
    num_test = len(all_samples_x)
    # test tensor
    x_test_tersor = torch.tensor(np_all_samples_x)
    y_test_tersor = torch.tensor(np_all_samples_y)

    test_dataset = Data.TensorDataset(x_test_tersor, y_test_tersor)

    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size)
    result_folder = "/home/wanghl/code/SFFN/HSFN_result"
    print("Outputing result")
    with open(os.path.join(result_folder, "SFFN_result2.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "real class", "predict"])
        model.val_start()
        acc = 0
        num_test = 0
        pred_ = []
        count = 0
        for i, val_data in enumerate(test_loader):
            # make sure val logger is used
            x, y = val_data
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            # print(type(y))
            # print(y.shape)
            pred = model.forward_pred(x)
            _, cls_result = torch.max(pred, 1)
            # print(cls_result.shape)
            for l in range(len(cls_result)):
                writer.writerow([all_files[count], all_samples_y[count], cls_result[l]])
                if all_samples_y[count] == 0:
                    if cls_result[l] == 0:
                        true_negative += 1
                    else:
                        false_positive += 1
                else:
                    if cls_result[l] == 1:
                        true_positive += 1
                    else:
                        false_negative += 1
                count = count + 1
            print(count)
        writer.writerow(["num of test files", len(all_files)])
        writer.writerow(["True Positive", true_positive])
        writer.writerow(["True Negative", true_negative])
        writer.writerow(["False Positive", false_positive])
        writer.writerow(["False Negative", false_negative])
        writer.writerow(["Accuracy", float(true_negative + true_positive) / len(all_files)])
        writer.writerow(["Precision", float(true_positive) / (true_positive + false_positive)])
        writer.writerow(["Recall", float(true_positive) / (true_positive + false_negative)])

        acc += torch.sum(cls_result == y)
        num_test += y.shape[0]

    # print(num_test)

    acc_ = acc.float() / num_test * 100
    print(acc_)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.iteritems()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.iteritems():
            tb_logger.log_value(prefix + k, v.val, step=step)


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='/home/wanghl/code/SFFN')
    # parser.add_argument('--root_path', default=r"E:\ngn\information_hiding\code_huyt\data", help='data root path')
    parser.add_argument('--language', default="ch", help='Language, ch or en')
    parser.add_argument('--code', default="g729a", help='coder')
    parser.add_argument('--algorithm', default="100_simul", help='embedding rate and steganography method')
    parser.add_argument('--time', default="1000", help='segment length (ms)')
    parser.add_argument('--num_epochs', default=50, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--num_layers', default=2, type=int,
                        help='Number of LSTM layers.')
    parser.add_argument('--input_dim1', default=3, type=int,
                        help='dimension of LSFCW.')
    parser.add_argument('--input_dim2', default=4, type=int,
                        help='dimension of PIF.')
    parser.add_argument('--hidden_size', default=50, type=int,
                        help='hidden_size of LSTM layers.')
    parser.add_argument('--n_class', default=2, type=int,
                        help='n_class.')
    parser.add_argument('--learning_rate', default=.001, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=1, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=100, type=int,
                        help='Number of steps to print and record the log.')
    # parser.add_argument('--num_val', default=1000, type=int, help='Number of steps to print and record the log.')
    opt = parser.parse_args()
    print("test...")
    print(opt)

    all_files = [(item, folder["class"]) for folder in test_FOLDERS for item in get_file_list(folder["folder"])]
    random.shuffle(all_files)
    model = MODEL(opt)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model_path = "/home/wanghl/code/SFFN/HSFN_result/ch_g729a_100_simul_1000ms/HSFN"
    best_checkpoint = torch.load(os.path.join(model_path, 'model_best.pth.tar'))
    print("load ok")
    model.load_state_dict(best_checkpoint['model'])
    best_epoch = best_checkpoint['epoch']
    best_acc = best_checkpoint['best_acc']
    acc_cover = best_checkpoint['acc_cover']
    acc_CNV = best_checkpoint['acc_CNV']
    acc_pitch = best_checkpoint['acc_pitch']
    test_validate(opt, all_files, model)
