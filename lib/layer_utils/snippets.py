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
import numpy as np
from layer_utils.generate_anchors import generate_anchors

#预处理生成anchor
def generate_anchors_pre(height, width, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
  shift_x = tf.range(width) * feat_stride # width 张量[0,16,32,...]
  shift_y = tf.range(height) * feat_stride # height 同上
  #tf.meshgrid给定 N 个一维坐标数组 *args，返回 N 维坐标数组的列表输出，用于计算 N 维网格上的表达式。
  shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
  # shift_x [[0,16,32]...]
  #          [0,16,32]...]
  #          .
  #          .
  #          .
  #          [0,16,32]...]]
  #shift_y [[0, 0, 0,...]
  #         [16,16,16,...]
  #         [32,32,32,...]
  #          .
  #          .
  #          .]      
  sx = tf.reshape(shift_x, shape=(-1,)) #转换为一维张量
  sy = tf.reshape(shift_y, shape=(-1,))
  # tf.stack将秩为 R 的张量列表堆叠成一个秩为 (R+1) 的张量
  # tf.transpose 交换矩阵维度
  shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))
  #shifts*T [[0,16,32,...,0,16,32,...]
  #         [0,0,0,...,16,16,16...]
  #         [0,16,32,...,0,16,32,...]
  #         [0,0,0,...,16,16,16...]]
  K = tf.multiply(width, height)
  shifts = tf.transpose(tf.reshape(shifts, shape=[1, K, 4]), perm=(1, 0, 2))
  # shifts.shape [K,1,4]
  # 生成anchor
  anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
  #A为anchor个数
  A = anchors.shape[0]
  #申明tf图中的常量
  anchor_constant = tf.constant(anchors.reshape((1, A, 4)), dtype=tf.int32)
  #anchor+偏移量
  #length为总anchor数量
  length = K * A
  anchors_tf = tf.reshape(tf.add(anchor_constant, shifts), shape=(length, 4))
  
  return tf.cast(anchors_tf, dtype=tf.float32), length

