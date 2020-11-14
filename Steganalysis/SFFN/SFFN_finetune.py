#!/usr/bin/env python
# -*-coding:utf-8 -*-
from __future__ import division
import numpy as np
import os, shutil, pickle
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
    {"class": 1, "folder": "/data/data_SFFN/g729a_Steg_QIM_feat"}
    # The folder that contains positive data files.

]
FOLDERS0 = [

    {"class": 0, "folder": "/data/data_SFFN/g729a_0_QIM_feat"}
    # The folder that contains negative data files.
]

FOLDERS3 = [
    {"class": 1, "folder": "/data/data_SFFN/g729a_Steg_PMS_feat"}
    # The folder that contains positive data files.

]
FOLDERS4 = [

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
            self.rnn1.cuda()
            self.rnn2.cuda()
            self.cnn.cuda()
            self.cls.cuda()
            self.cls1.cuda()
            self.cls2.cuda()

        self.criterion = SimpleLoss(opt.n_class)
        self.params = list(self.cnn.parameters()) + list(self.cls.parameters()) + list(self.rnn1.parameters()) + list(
            self.rnn2.parameters())
        self.params_other = list(self.cls1.parameters()) + list(self.cls2.parameters())

        for param in self.params_other:
            param.requires_grad = False
        self.optimizer = optim.Adam(self.params, lr=opt.learning_rate)
        self.Eiters = 0

    def load_state_dict_pretrain1(self, state_dict):
        self.rnn1.load_state_dict(state_dict[0])
        self.cls1.load_state_dict(state_dict[1])

    def load_state_dict_pretrain2(self, state_dict):
        self.rnn2.load_state_dict(state_dict[0])
        self.cls2.load_state_dict(state_dict[1])

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
        x1 = x[:, :, 0:opt.input_dim1]
        x2 = x[:, :, opt.input_dim1:opt.input_dim1 + opt.input_dim2]
        embed1, _ = self.rnn1(x1)
        embed2, _ = self.rnn2(x2)
        embed1_t = embed1[:, -1, :]
        embed2_t = embed2[:, -1, :]
        embed_cat = torch.stack((embed1, embed2), 1)
        embed = self.cnn(embed_cat)
        pred = self.cls(embed)
        pred1 = self.cls1(embed1_t)
        pred2 = self.cls2(embed2_t)
        return pred, pred1, pred2

    def train_emb(self, x, y):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        pred, _, _ = self.forward_pred(x)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(pred, y)
        self.logger.update('loss', loss.data.item())
        # compute gradient and do SGD step
        loss.backward()
        self.optimizer.step()


def get_loaders(opt):
    train_positive_QIM_files = [(item, folder["class"]) for folder in FOLDERS1 for item in
                                get_file_list(folder["folder"])]
    train_positive_LSFCW = [(parse_sample(item[0])) for item in train_positive_QIM_files]

    train_negative_QIM_files = [(item, folder["class"]) for folder in FOLDERS0 for item in
                                get_file_list(folder["folder"])]
    train_negative_LSFCW = [(parse_sample(item[0])) for item in train_negative_QIM_files]

    train_positive_PMS_files = [(item, folder["class"]) for folder in FOLDERS3 for item in
                                get_file_list(folder["folder"])]
    train_positive_PIF = [(parse_sample(item[0])) for item in train_positive_PMS_files]

    train_negative_PMS_files = [(item, folder["class"]) for folder in FOLDERS4 for item in
                                get_file_list(folder["folder"])]
    train_negative_PIF = [(parse_sample(item[0])) for item in train_negative_PMS_files]

    train_positive_LSFCW = np.array(train_positive_LSFCW).astype('float32')
    train_negative_LSFCW = np.array(train_negative_LSFCW).astype('float32')
    train_positive_PIF = np.array(train_positive_PIF).astype('float32')
    train_negative_PIF = np.array(train_negative_PIF).astype('float32')

    train_CNV = np.concatenate((train_positive_LSFCW, train_negative_PIF), axis=2)
    train_pitch = np.concatenate((train_negative_LSFCW, train_positive_PIF), axis=2)
    train_cover = np.concatenate((train_negative_LSFCW, train_negative_PIF), axis=2)
    print("train_CNV", train_CNV.shape)
    print("train_pitch", train_pitch.shape)
    print("train_cover", train_cover)
    train_positive_LSFCW = []
    train_negative_LSFCW = []
    train_positive_PIF = []
    train_negative_PIF = []

    # spliting validation data
    num_val = int(len(train_CNV) / 5)
    opt.num_val = num_val
    val_CNV = train_CNV[-opt.num_val:, :, :]
    val_pitch = train_pitch[-opt.num_val:, :, :]
    val_cover = train_cover[-opt.num_val:, :, :]
    print('num_val', val_cover.shape[0])

    train_CNV = train_CNV[:-opt.num_val, :, :]
    train_pitch = train_pitch[:-opt.num_val, :, :]
    train_cover = train_cover[:-opt.num_val, :, :]
    num_train = train_cover.shape[0]
    opt.num_train = num_train
    print('num_train', num_train)

    x_train = np.concatenate((train_CNV, train_pitch, train_cover, train_cover), axis=0)
    y_train = np.concatenate((np.ones((num_train * 2,), dtype=np.int), np.zeros((num_train * 2,), dtype=np.int)),
                             axis=0)

    train_CNV = []
    train_pitch = []
    train_cover = []

    x_train_tersor = torch.tensor(x_train)
    y_train_tersor = torch.tensor(y_train)

    val_cover_tersor = torch.tensor(val_cover)
    val_CNV_tersor = torch.tensor(val_CNV)
    val_pitch_tersor = torch.tensor(val_pitch)
    y_val_cover_tersor = torch.tensor(np.zeros(opt.num_val, ), dtype=torch.int64)
    y_val_positive_tersor = torch.tensor(np.ones(opt.num_val, ), dtype=torch.int64)

    # TensorDataset
    train_dataset = Data.TensorDataset(x_train_tersor, y_train_tersor)
    # test_cover_dataset = Data.TensorDataset(test_cover_tersor, y_test_cover_tersor)
    # test_CNV_dataset = Data.TensorDataset(test_CNV_tersor, y_test_positive_tersor)
    # test_pitch_dataset = Data.TensorDataset(test_pitch_tersor, y_test_positive_tersor)
    # test_alter_dataset = Data.TensorDataset(test_alter_tersor, y_test_positive_tersor)
    val_cover_dataset = Data.TensorDataset(val_cover_tersor, y_val_cover_tersor)
    val_CNV_dataset = Data.TensorDataset(val_CNV_tersor, y_val_positive_tersor)
    val_pitch_dataset = Data.TensorDataset(val_pitch_tersor, y_val_positive_tersor)

    # DataLoader
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
    # test_cover_loader = Data.DataLoader(dataset=test_cover_dataset, batch_size=opt.batch_size)
    # test_CNV_loader = Data.DataLoader(dataset=test_CNV_dataset, batch_size=opt.batch_size)
    # test_pitch_loader = Data.DataLoader(dataset=test_pitch_dataset, batch_size=opt.batch_size)
    # test_alter_loader = Data.DataLoader(dataset=test_alter_dataset, batch_size=opt.batch_size)
    val_cover_loader = Data.DataLoader(dataset=val_cover_dataset, batch_size=opt.batch_size)
    val_CNV_loader = Data.DataLoader(dataset=val_CNV_dataset, batch_size=opt.batch_size)
    val_pitch_loader = Data.DataLoader(dataset=val_pitch_dataset, batch_size=opt.batch_size)

    return train_loader, val_cover_loader, val_CNV_loader, val_pitch_loader


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
    acc1 = 0
    acc2 = 0
    num_test = 0
    pred_ = []
    pred1_ = []
    pred2_ = []

    for i, val_data in enumerate(val_loader):
        # make sure val logger is used
        x, y = val_data
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        pred, pred1, pred2 = model.forward_pred(x)
        _, cls_result = torch.max(pred, 1)
        _, cls_result1 = torch.max(pred1, 1)
        _, cls_result2 = torch.max(pred2, 1)
        pred_.append(pred[:, 1])
        pred1_.append(pred1[:, 1])
        pred2_.append(pred2[:, 1])
        acc += torch.sum(cls_result == y)
        acc1 += torch.sum(cls_result1 == y)
        acc2 += torch.sum(cls_result2 == y)
        num_test += y.shape[0]
    acc_ = acc.float() / num_test * 100
    acc1_ = acc1.float() / num_test * 100
    acc2_ = acc2.float() / num_test * 100
    return acc_, acc1_, acc2_, pred_, pred1_, pred2_


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
    parser.add_argument('--num_epochs', default=30, type=int,
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
    print(opt)

    result_root_path = os.path.join(opt.root_path, "HSFN_result")
    if not os.path.exists(result_root_path):
        os.mkdir(result_root_path)
    result_path_ = os.path.join(result_root_path, "%s_%s_%s_%sms" % (opt.language, opt.code, opt.algorithm, opt.time))
    if not os.path.exists(result_path_):
        os.mkdir(result_path_)
    result_path = os.path.join(result_path_, "HSFN_finetune")
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(result_path, flush_secs=5)

    # Construct the model
    model = MODEL(opt)

    print("pretrained model loading...")
    for pretrain_path in ["HSFN"]:
        model_path = os.path.join(result_path_, pretrain_path, "model_best.pth.tar")
        if os.path.exists(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model'])
        else:
            print("=> no checkpoint found at '{}'".format(model_path))
            exit()

    # Load data loaders
    print("data loading...")
    train_loader, val_cover_loader, val_CNV_loader, val_pitch_loader = get_loaders(
        opt)

    acc_cover, acc_cover1, acc_cover2, _, _, _ = validate(opt, val_cover_loader, model)
    acc_CNV, acc_CNV1, acc_CNV2, _, _, _ = validate(opt, val_CNV_loader, model)
    acc_pitch, acc_pitch1, acc_pitch2, _, _, _ = validate(opt, val_pitch_loader, model)
    logging.info("acc_cover: %.2f \t acc_CNV: %.2f \t acc_pitch: %.2f" % (acc_cover, acc_CNV, acc_pitch))
    logging.info("acc_cover1: %.2f \t acc_CNV1: %.2f \t acc_pitch1: %.2f" % (acc_cover1, acc_CNV1, acc_pitch1))
    logging.info("acc_cover2: %.2f \t acc_CNV2: %.2f \t acc_pitch2: %.2f" % (acc_cover2, acc_CNV2, acc_pitch2))

    best_acc = 0
    print("training begin...")
    for epoch in range(opt.num_epochs):
        # train for one epoch
        train(opt, train_loader, model, epoch)

        # evaluate on validation set
        # val_cover_loader, val_CNV_loader, val_pitch_loader
        acc_cover, _, _, _, _, _ = validate(opt, val_cover_loader, model)
        acc_CNV, _, _, _, _, _ = validate(opt, val_CNV_loader, model)
        acc_pitch, _, _, _, _, _ = validate(opt, val_pitch_loader, model)

        val_logger = LogCollector()
        model.logger = val_logger
        logging.info("acc_cover: %.2f \t acc_CNV: %.2f \t acc_pitch: %.2f \t acc_average: %.2f" % (
            acc_cover, acc_CNV, acc_pitch, (acc_cover + acc_CNV + acc_pitch) / 3))

        # record metrics in tensorboard
        tb_logger.log_value('acc_cover', acc_cover, step=model.Eiters)
        tb_logger.log_value('acc_CNV', acc_CNV, step=model.Eiters)
        tb_logger.log_value('acc_pitch', acc_pitch, step=model.Eiters)

        acc = (acc_cover + acc_CNV + acc_pitch) / 3
        # remember best accuracy and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_acc': best_acc,
            'acc_cover': acc_cover,
            'acc_CNV': acc_CNV,
            'acc_pitch': acc_pitch,
            'Eiters': model.Eiters,
        }, is_best, prefix=result_path + '/')
    # write result
    best_checkpoint = torch.load(os.path.join(result_path, 'model_best.pth.tar'))
    model.load_state_dict(best_checkpoint['model'])
    best_epoch = best_checkpoint['epoch']
    best_acc = best_checkpoint['best_acc']
    acc_cover = best_checkpoint['acc_cover']
    acc_CNV = best_checkpoint['acc_CNV']
    acc_pitch = best_checkpoint['acc_pitch']

    print("loaded best_checkpoint (epoch %d, best_acc %.2f, acc_cover %.2f, acc_CNV %.2f,acc_pich %.2f)" % (
        best_epoch, best_acc, acc_cover, acc_CNV, acc_pitch))

    f = open(os.path.join(result_path, "result_HSFN_without_test.txt"), 'w')
    f.write("num_validation: %d \t num_train: %d \n" % (opt.num_val, opt.num_train))
    f.write("loaded best_checkpoint (epoch %d, best_acc %.2f, acc_cover %.2f, acc_cnv %.2f,acc_pitch %.2f)\n" % (
        best_epoch, best_acc, acc_cover, acc_CNV, acc_pitch))
    f.close()

