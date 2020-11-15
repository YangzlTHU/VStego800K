#!/usr/bin/env python
# -*-coding:utf-8 -*-
from __future__ import division
import numpy as np
import os, shutil
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

FOLDERS1 = [
    {"class": 1, "folder": "/data/data_SFFN/g729a_Steg_PMS_feat"},
    # The folder that contains positive data files.

]
FOLDERS0 = [

    {"class": 0, "folder": "/data/data_SFFN/g729a_0_PMS_feat"}
    # The folder that contains negative data files.
]


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


class SimpleLoss(nn.Module):

    def __init__(self, n_class):
        super(SimpleLoss, self).__init__()
        self.n_class = n_class

    def forward(self, pred, label):
        # one-hot
        label_onehot = torch.FloatTensor(label.size()[0], self.n_class)
        if torch.cuda.is_available():
            label = label.cuda()
            label_onehot = label_onehot.cuda()
        label_onehot.zero_()
        label_onehot.scatter_(1, torch.unsqueeze(label, 1), 1)
        loss_main = F.binary_cross_entropy(pred, label_onehot)
        return loss_main


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
            self.rnn2.cuda()
            self.cls2.cuda()

        self.criterion = SimpleLoss(opt.n_class)
        self.params = list(self.rnn2.parameters()) + list(self.cls2.parameters())
        self.params_other = list(self.rnn1.parameters()) + list(self.cls1.parameters()) + list(
            self.cnn.parameters()) + list(self.cls.parameters())

        for param in self.params_other:
            param.requires_grad = False

        self.optimizer = optim.Adam(self.params, lr=opt.learning_rate)
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.rnn2.state_dict(), self.cls2.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.rnn2.load_state_dict(state_dict[0])
        self.cls2.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.rnn1.train()
        self.cls1.train()
        self.rnn2.train()
        self.cls2.train()
        self.cnn.train()
        self.cls.train()

        for param in self.params:
            param.requires_grad = True

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

    def forward_loss(self, pred, label):
        loss = self.criterion(pred, label)
        return loss

    def forward_pred(self, x):
        rnn_out, _ = self.rnn2(x)
        embed = rnn_out[:, -1, :]  # output of the last time
        pred = self.cls2(embed)
        return pred

    def train_emb(self, x, y):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        pred = self.forward_pred(x)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(pred, y)
        self.logger.update('loss', loss.data.item())
        # compute gradient and do SGD step
        loss.backward()
        self.optimizer.step()


def get_loaders(opt):
    # experiment_data_path = os.path.join(opt.root_path, "experiment_data")

    # load training data
    train_positive_files = [(item, folder["class"]) for folder in FOLDERS1 for item in get_file_list(folder["folder"])]
    train_positive = [(parse_sample(item[0])) for item in train_positive_files]

    train_negative_files = [(item, folder["class"]) for folder in FOLDERS0 for item in get_file_list(folder["folder"])]
    train_negative = [(parse_sample(item[0])) for item in train_negative_files]
    num_val = int(len(train_negative_files) / 5)
    opt.num_val = num_val
    train_positive = np.array(train_positive).astype('float32')
    train_negative = np.array(train_negative).astype('float32')

    val_positive = train_positive[-opt.num_val:, :, :]
    val_negative = train_negative[-opt.num_val:, :, :]
    print('num_val', num_val)

    train_positive = train_positive[:-opt.num_val, :, :]
    train_negative = train_negative[:-opt.num_val, :, :]
    num_train = train_positive.shape[0]
    opt.num_train = num_train
    print('num_train', num_train)

    x_train = np.concatenate((train_positive, train_negative), axis=0)
    y_train = np.concatenate((np.ones((num_train,), dtype=np.int), np.zeros((num_train,), dtype=np.int)), axis=0)

    x_train_tersor = torch.tensor(x_train)
    y_train_tersor = torch.tensor(y_train)

    # val tensor
    val_negative_tersor = torch.tensor(val_negative)
    val_positive_tersor = torch.tensor(val_positive)
    y_val_negative_tersor = torch.tensor(np.zeros(opt.num_val, ), dtype=torch.int64)
    y_val_positive_tersor = torch.tensor(np.ones(opt.num_val, ), dtype=torch.int64)

    # TensorDataset
    train_dataset = Data.TensorDataset(x_train_tersor, y_train_tersor)
    val_negative_dataset = Data.TensorDataset(val_negative_tersor, y_val_negative_tersor)
    val_positive_dataset = Data.TensorDataset(val_positive_tersor, y_val_positive_tersor)

    # DataLoader
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_negative_loader = Data.DataLoader(dataset=val_negative_dataset, batch_size=opt.batch_size)
    val_positive_loader = Data.DataLoader(dataset=val_positive_dataset, batch_size=opt.batch_size)

    return train_loader, val_negative_loader, val_positive_loader


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


def validate(opt, val_loader, model):
    # compute the predictions
    model.val_start()
    acc = 0
    num_test = 0

    for i, val_data in enumerate(val_loader):
        # make sure val logger is used
        x, y = val_data
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        pred = model.forward_pred(x)
        _, cls_result = torch.max(pred, 1)
        acc += torch.sum(cls_result == y)
        num_test += y.shape[0]
    acc_ = acc.float() / num_test * 100
    return acc_


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


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
                        help='dimension of PIF.')
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
    print("rnn2")
    print(opt)

    result_root_path = os.path.join(opt.root_path, "HSFN_result")
    if not os.path.exists(result_root_path):
        os.mkdir(result_root_path)
    result_path = os.path.join(result_root_path, "%s_%s_%s_%sms" % (opt.language, opt.code, opt.algorithm, opt.time))
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    result_path = os.path.join(result_path, "rnn2")
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(result_path, flush_secs=5)

    # Construct the model
    model = MODEL(opt)

    # Load data loaders
    print("data loading...")
    train_loader, val_negative_loader, val_positive_loader = get_loaders(opt)

    best_acc = 0
    print("training begin...")
    for epoch in range(opt.num_epochs):
        # train for one epoch
        train(opt, train_loader, model, epoch)

        # evaluate on validation set
        # val_negative_loader, val_positive_loader
        acc_negative = validate(opt, val_negative_loader, model)
        acc_positive = validate(opt, val_positive_loader, model)

        val_logger = LogCollector()
        model.logger = val_logger
        logging.info("acc_cover: %.2f \t acc_positive: %.2f" % (acc_negative, acc_positive))

        # record metrics in tensorboard
        tb_logger.log_value('acc_cover', acc_negative, step=model.Eiters)
        tb_logger.log_value('acc_positive', acc_positive, step=model.Eiters)

        acc = (acc_negative + acc_positive) / 2
        # remember best accuracy and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_acc': best_acc,
            'acc_cover': acc_negative,
            'acc_positive': acc_positive,
            'Eiters': model.Eiters,
        }, is_best, prefix=result_path + '/')

    # write result
    best_checkpoint = torch.load(os.path.join(result_path, 'model_best.pth.tar'))
    model.load_state_dict(best_checkpoint['model'])
    best_epoch = best_checkpoint['epoch']
    best_acc = best_checkpoint['best_acc']
    acc_negative = best_checkpoint['acc_cover']
    acc_positive = best_checkpoint['acc_positive']

    print("loaded best_checkpoint (epoch %d, best_acc %.2f, acc_cover %.2f, acc_positive %.2f)" % (
        best_epoch, best_acc, acc_negative, acc_positive))

    f = open(os.path.join(result_path, "result.txt"), 'w')
    f.write("num_validation: %d \t num_train: %d \n" % (opt.num_val, opt.num_train))
    f.write("loaded best_checkpoint (epoch %d, best_acc %.2f, acc_cover %.2f, acc_positive %.2f)\n" % (
        best_epoch, best_acc, acc_negative, acc_positive))
    f.close()
