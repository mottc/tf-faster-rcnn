#-*-coding:utf-8 -*-
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

from nets.network import Network
from model.config import cfg

#VGG16网络结构
class vgg16(Network):
  def __init__(self):
    Network.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._scope = 'vgg_16'

  def _image_to_head(self, is_training, reuse=None):
    #image-head网络
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      #slim.repeat重复操作
      #两个卷积层（核大小3*3，个数64）
      net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],
                          trainable=False, scope='conv1')
      #2*2池化层
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
      
      #两个卷积层（核大小3*3，个数128）      
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                        trainable=False, scope='conv2')
      #2*2池化层
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')

      #三个卷积层（核大小3*3，个数256） 
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                        trainable=is_training, scope='conv3')
      #2*2池化层
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')

      #三个卷积层（核大小3*3，个数512）
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv4')
      #2*2池化层
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
      
      #三个卷积层（核大小3*3，个数512）
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv5')

    #保存网络图
    self._act_summaries.append(net)
    self._layers['head'] = net
    
    return net

  # 构建头尾网络图
  def _head_to_tail(self, pool5, is_training, reuse=None):
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      # 平整输入（reshape）
      pool5_flat = slim.flatten(pool5, scope='flatten')
      # fc6全连接层，连接到4096神经元
      fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
      # 若用于训练则添加dropout层
      if is_training:
        fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, 
                            scope='dropout6')
      # fc7全连接层,连接到4096神经元
      fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
      # 若用于训练则添加dropout层
      if is_training:
        fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, 
                            scope='dropout7')

    return fc7

  # 获取用于保存的变量
  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the conv weights that are fc weights in vgg16
      # 排除VGG全连接层中的卷积层变量
      if v.name == (self._scope + '/fc6/weights:0') or \
         v.name == (self._scope + '/fc7/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      # exclude the first conv layer to swap RGB to BGR
      # 排除第一个用于RGB到BGR转换的卷积层
      if v.name == (self._scope + '/conv1/conv1_1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      # 若变量名位于var_keep_dic则加入保存变量
      if v.name.split(':')[0] in var_keep_dic:
        print('Variables restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  # 修复变量
  def fix_variables(self, sess, pretrained_model):
    print('Fix VGG16 layers..')
    with tf.variable_scope('Fix_VGG16') as scope:
      with tf.device("/cpu:0"):
        # fix the vgg16 issue from conv weights to fc weights
        # 将vgg16中的卷积层变量转换为全连接层变量
        # fix RGB to BGR
        # 将RGB转换为BGR
        
        fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
        fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
        conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
        # 建立存储器
        restorer_fc = tf.train.Saver({self._scope + "/fc6/weights": fc6_conv, 
                                      self._scope + "/fc7/weights": fc7_conv,
                                      self._scope + "/conv1/conv1_1/weights": conv1_rgb})
        # 读取预训练模型
        restorer_fc.restore(sess, pretrained_model)
        # 修改变量
        sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc6/weights:0'], tf.reshape(fc6_conv, 
                            self._variables_to_fix[self._scope + '/fc6/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc7/weights:0'], tf.reshape(fc7_conv, 
                            self._variables_to_fix[self._scope + '/fc7/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/conv1_1/weights:0'], 
                            tf.reverse(conv1_rgb, [2])))
