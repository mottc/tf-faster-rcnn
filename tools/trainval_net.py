# -*- coding:utf-8 -*-  
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import #绝对引入
from __future__ import division #精确除法
from __future__ import print_function #print函数

import _init_paths
from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.factory import get_imdb
import datasets.imdb
import argparse #命令解析库
import pprint #打印模块
import numpy as np
import sys

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

def parse_args():#获取命令转换为参数
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str) #配置文件
  parser.add_argument('--weight', dest='weight',
                      help='initialize with pretrained model weights',
                      type=str) #（预训练的）初始化权值
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='voc_2007_trainval', type=str) #数据集名称
  parser.add_argument('--imdbval', dest='imdbval_name',
                      help='dataset to validate on',
                      default='voc_2007_test', type=str) #测试集名称
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=70000, type=int) #迭代轮数
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str) #模型标签(?)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152, mobile',
                      default='res50', type=str) #网络结构
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER) #设置配置关键字(?)
  ## 如果sys.argv长度为1,则说明没有参数传入，系统会退出
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


def combined_roidb(imdb_names):#整合多个roidb（如果有的话）
  ## 多个数据集名称以‘+’连接
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    ## imdb为存在一个字典(easydict)里的pascal_voc类的一个对象，e.g.{voc_2007_train:内容，voc_2007_val:内容，voc_2007_test:内容,voc_2007_test:内容,voc_2012_train:内容...}
    ## 内容里有该类里的各种self名称与操作，包括roi信息等等
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    ## __C.TRAIN.PROPOSAL_METHOD = 'gt'
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    ## get_training_roidb函数返回imdb对象的各种roi与图片信息，用于训练
    ## 这是一个列表，列表中存的是各个图片的字典，字典中存roi信息，字典引索为图片引索
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')] #从数据集获取roi集
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)
  return imdb, roidb


if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)
  ## 如果还有其他配置文件，就加载
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  
  #设置随机种子
  np.random.seed(cfg.RNG_SEED)

  # train set 数据集
  imdb, roidb = combined_roidb(args.imdb_name)
  print('{:d} roidb entries'.format(len(roidb)))

  # output directory where the models are saved 模型保存路径
  output_dir = get_output_dir(imdb, args.tag)
  print('Output will be saved to `{:s}`'.format(output_dir))

  # tensorboard directory where the summaries are saved during training 训练总结保存路径
  tb_dir = get_output_tb_dir(imdb, args.tag)
  print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

  # also add the validation set, but with no flipping images
  # 载入验证集，但不使用翻转图像
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  _, valroidb = combined_roidb(args.imdbval_name)
  print('{:d} validation roidb entries'.format(len(valroidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip

  # load network 载入模型
  if args.net == 'vgg16':
    net = vgg16()
  elif args.net == 'res50':
    net = resnetv1(num_layers=50)
  elif args.net == 'res101':
    net = resnetv1(num_layers=101)
  elif args.net == 'res152':
    net = resnetv1(num_layers=152)
  elif args.net == 'mobile':
    net = mobilenetv1()
  else:
    raise NotImplementedError

  #训练网络
  train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
            pretrained_model=args.weight,
            max_iters=args.max_iters)
