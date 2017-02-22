import caffe
import numpy as np
from numpy import linalg as LA
import argparse
import pprint
import scipy.io as scio
import math
import os


def Programpause():
    inputstr = raw_input('Press e to exit, Press other key to continue!!!\n')
    if (inputstr == 'e'):
        os._exit(0)


class Set2SetWithLossLayer(caffe.Layer):

    @classmethod
    def parse_args(cls, argsStr):
        parser = argparse.ArgumentParser(description='Set2SetLossLayer')
        parser.add_argument('--rho', default=0.1, type=float)
        parser.add_argument('--phase', default='', type=str)
        parser.add_argument('--datafile', default='', type=str)
        parser.add_argument('--difffile', default='', type=str)
        parser.add_argument('--avglossfile', default='', type=str)
        parser.add_argument('--test_interval', default=0, type=int)
        args = parser.parse_args(argsStr.split())
        print('Using Config:')
        pprint.pprint(args)
        return args

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        self.params_ = Set2SetWithLossLayer.parse_args(self.param_str)
        self.rho = self.params_.rho
        self.phase = self.params_.phase
        self.icount = 1
        self.zerocount = 0
        self.lcount = 0
        self.sumloss = 0.0
        self.avgcount = 0
        self.filename_data = self.params_.datafile
        self.filename_diff = self.params_.difffile

    def reshape(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        else:
            if (bottom[0].data.shape[0] != bottom[1].data.shape[0]):
                raise Exception('Two inputs need to have same shape!!!')
        self.num = bottom[0].data.shape[0]
        self.channel = bottom[0].data.shape[1]
        top[0].reshape(1)

    def find_same(self, bottom, top):
        buffersize = bottom[1].shape[0]
        same_index = -np.ones(buffersize, dtype=int)
        for bs1 in range(0, buffersize):
            for bs2 in range(0, buffersize):
                if (bottom[1].data[bs1] == bottom[1].data[bs2] and bs1 != bs2):
                    same_index[bs1] = bs2
                    break
        return same_index

    def forward(self, bottom, top):
        loss = 0
        loss1 = 0
        loss2 = 0
        self.bottom_data = bottom[0].data
        self.diff_matrix = np.zeros((self.num, self.num))
        for xi in range(0, self.num):
            for xj in range(0, self.num):
                self.diff_matrix[xi, xj] = (LA.norm(
                    bottom[0].data[xi] - bottom[0].data[xj]))
        Na = self.num * self.num
        self.sameindex = self.find_same(bottom, top)
        # print 'same index: ', self.sameindex
        # Programpause()
        self.Ns = 0
        for x in self.sameindex:
            if (x > -1):
                self.Ns = self.Ns + 1
        self.Nd = Na - self.Ns
        self.diff_matrix_same = np.zeros(self.Ns)
        si = 0
        for i in range(0, self.num):
            if (self.sameindex[i] > -1):
                self.diff_matrix_same[si] = self.diff_matrix[i, self.sameindex[i]]
                si = si + 1
        loss1 = np.sum(self.diff_matrix_same * self.diff_matrix_same)
        loss2 = np.sum(self.diff_matrix ** 2) - loss1
        if (self.Ns > 0):
            loss = (loss1 / self.Ns) - self.rho * (loss2 / self.Nd)
        else:
            loss = 0 - self.rho * (loss2 / self.Nd)
        top[0].data[...] = loss
        if (self.params_.phase == 'train'):
            self.sumloss = self.sumloss + loss
            self.lcount = self.lcount + 1
            if (self.lcount == self.params_.test_interval):
                avgloss = self.sumloss / self.lcount
                self.lcount = 0
                self.avgcount = self.avgcount + 1
                f = open(self.params_.avglossfile, 'a')
                f.write(str(self.avgcount) + '   ' + str(avgloss) + '\n')
                f.close()
                self.sumloss = 0.0
        else:
            print 'bottom data:\n', bottom[0].data
        if (loss == 0):
            self.zerocount = self.zerocount + 1
            if (self.zerocount > 20):
                print 'bottom data:\n', bottom[0].data
        else:
            self.zerocount = 0
        # print 'bottom data:\n', bottom[0].data
        f = open(self.filename_data, 'a')
        f.write(str(self.icount) + '   ' + str(loss) + '\n')
        f.close()
        self.icount += 1
        # raise Exception('Pause!!!')

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            diff = np.zeros((self.num, self.channel))
            for person in range(0, self.num):
                if (self.sameindex[person] > -1):
                    for fea_ct in range(0, self.channel):
                        d1 = bottom[0].data[person, fea_ct] - bottom[0].data[self.sameindex[person], fea_ct]
                        d2 = 0 - d1
                        for person2 in range(0, self.num):
                            d2 = d2 + bottom[0].data[person, fea_ct] - bottom[0].data[person2, fea_ct]
                        diff[person, fea_ct] = 4 * d1 / self.Ns - 4 * self.rho * d2 / self.Nd
                        if (math.isnan(diff[person, fea_ct])):
                            raise Exception('Pause!!!')
                else:
                    for fea_ct in range(0, self.channel):
                        d1 = 0
                        d2 = 0
                        for person2 in range(0, self.num):
                            d2 = d2 + bottom[0].data[person, fea_ct] - bottom[0].data[person2, fea_ct]
                        diff[person, fea_ct] = 0 - 4 * self.rho * d2 / self.Nd
                        if (math.isnan(diff[person, fea_ct])):
                            raise Exception('Pause!!!')
            bottom[i].diff[...] = diff
            # print 'bottom diff:\n', bottom[i].diff
            # Programpause()
            if (self.zerocount > 20):
                print 'bottom diff:\n', bottom[i].diff
                Programpause()
