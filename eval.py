# -----------------------------------------------------------------------
# eval.py
# Verification in Python code
#
# Creation Date   : 04/Aug./2017
# Copyright (C) <2017> Hiroki Nakahara, All rights reserved.
# 
# Released under the GPL v2.0 License.
# 
# Acknowledgements:
# This source code is based on following projects:
#
# Chainer binarized neural network by Daisuke Okanohara
# https://github.com/hillbig/binary_net
# Various CNN models including Deep Residual Networks (ResNet) 
#  for CIFAR10 with Chainer by mitmul
# https://github.com/mitmul/chainer-cifar10
# -----------------------------------------------------------------------

import argparse
#import cPickle as pickle # python 2.7
import _pickle as pickle # python 3.5
import numpy as np
import os
import chainer
from chainer import optimizers, Variable
from chainer import serializers
import net3 # it will be generated by the GUINNESS

import trainer
import chainer.links as L

import time
import weight_clip

import cv2
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation Python Code')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU device ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', type=str, default='temp.model',
                        help='Pre-Trained Model Name')
    parser.add_argument('--dataset', '-d', type=str, default='two96_dataset.pkl',
                        help='Dataset image pkl file path')
    parser.add_argument('--size', type=int, default=32,
                        help='Test Image Size')
    parser.add_argument('--testnum', type=int, default=10,
                        help='Test Image Num')
                        
    args = parser.parse_args()

    print('loading dataset...')
    fname = args.dataset + '_dataset.pkl'
    with open(fname, 'rb') as f:
        images = pickle.load(f)
        threshold = np.int32(len(images['train'])/10*9)
        train_x = images['train'][:threshold].astype(np.float32)
        valid_x = images['train'][threshold:].astype(np.float32)
        test_x = images['test'].astype(np.float32)

    fname = args.dataset + '_label.pkl'
    with open(fname, 'rb') as f:
        labels = pickle.load(f)
        train_y = labels['train'][:threshold].astype(np.int32)
        valid_y = labels['train'][threshold:].astype(np.int32)
        test_y = labels['test'].astype(np.int32)

    print('start evaluation')

    net = net3.CNN()
    print("load pre-trained npz")
    serializers.load_npz(args.model, net)

    # set image size
    img_siz = args.size

    eval_x = np.ones((1,3,img_siz,img_siz))

    # load tag file
    name = []
    fname = args.dataset + '_tag.txt' # tag file be generated by 'gen_training_data.py'
    with open(fname, 'r') as f:
        lines2 = f.readlines()
        for line in lines2:
            name.append(line.rstrip('\n\r'))

    n_class = len(name)

    conf_matrix = np.zeros((n_class,n_class))

    # specify the number of tests
    if(args.testnum == -1):
        n_tests = len(test_x)
    else:
        n_tests = args.testnum
    n_acc   = 0

    # perform test
    for idx in range(0,n_tests):
        image = test_x
        image = image.clip(0,255).astype(np.uint8)

        print("label=%d(%s)" % (test_y[idx],name[test_y[idx]]))

        # Note that, the test image is generated by the OpenCV2.0, thus, its format consists of 'BGR' not 'RGB'
        image1 = image[idx].reshape(3, img_siz, img_siz).transpose(1, 2, 0)

# generate test bench
# you can comment out following to generate more test bech for C/C++ simulation in the Vivado HLS, and an FPGA board
#        '''
        bench_img = image1.reshape(-1,)
        if(n_tests==-1):
            fname = 'test_img_%d.txt' % idx # + str(idx) + '.txt'
            print(' Test Image Fileout -> %s' % fname)
            np.savetxt(fname, bench_img, fmt="%.0f", delimiter=",")

        if idx == 0:
            fname = 'HLS/test_img_%d.txt' % idx # + str(idx) + '.txt'
            print(' Test Image Fileout -> %s' % fname)
            np.savetxt(fname, bench_img, fmt="%.0f", delimiter=",")
#        '''

        eval_x[0,:,:,:] = test_x[idx] #/ 256.0

        result = net(Variable(eval_x.astype(np.float32)),train=False)
        print(result.data)
        print("test=%d(%s)" % (result.data.argmax(),name[result.data.argmax()]))
        if(n_tests==-1):
            pil_img = Image.fromarray(image1)
            pil_img.save('test_img_%d.png'  %idx)

        # regist a confusion matrix
        conf_matrix[test_y[idx],result.data.argmax()] = conf_matrix[test_y[idx],result.data.argmax()] + 1

        if test_y[idx] == result.data.argmax():
            n_acc = n_acc + 1

    # show a confusion matrix    
    print("Confusion Matrix")
    print(conf_matrix.astype(np.int32))
    print("# corrests=%d" % n_acc)
    print("Accuracy=%f" % (float(n_acc) / n_tests))

# -----------------------------------------------------------------------
# END OF PROGRAM
# -----------------------------------------------------------------------