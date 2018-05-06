#-*- coding:utf-8 -*-
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen and Zheqi He
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
try:
  import cPickle as pickle
except ImportError:
  import pickle
import numpy as np
import os
import sys
import glob
import time

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

#解决方案封装类
class SolverWrapper(object):
  """
    A wrapper class for the training process
  """

  def __init__(self, sess, network, imdb, roidb, valroidb, output_dir, tbdir, pretrained_model=None):
    self.net = network
    self.imdb = imdb
    self.roidb = roidb
    self.valroidb = valroidb
    self.output_dir = output_dir
    self.tbdir = tbdir
    # Simply put '_val' at the end to save the summaries from the validation set
    self.tbvaldir = tbdir + '_val'
    if not os.path.exists(self.tbvaldir):
      os.makedirs(self.tbvaldir)
    self.pretrained_model = pretrained_model

  def snapshot(self, sess, iter):
    # 保存模型快照
    net = self.net

    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    # Store the model snapshot
    filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt'
    filename = os.path.join(self.output_dir, filename)
    self.saver.save(sess, filename)
    print('Wrote snapshot to: {:s}'.format(filename))

    # Also store some meta information, random state, etc.
    nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
    nfilename = os.path.join(self.output_dir, nfilename)
    # current state of numpy random
    # 当前的随机状态
    st0 = np.random.get_state()
    # current position in the database
    # 数据库中的当前位置
    cur = self.data_layer._cur
    # current shuffled indexes of the database
    # 洗后的当前位置索引
    perm = self.data_layer._perm
    # current position in the validation database
    cur_val = self.data_layer_val._cur
    # current shuffled indexes of the validation database
    perm_val = self.data_layer_val._perm

    # Dump the meta info
    # 保存信息
    with open(nfilename, 'wb') as fid:
      pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

    return filename, nfilename


  def from_snapshot(self, sess, sfile, nfile):
    #从保存的snapshot快照中读取信息
    #读取checkpoint模型文件的信息
    print('Restoring model snapshots from {:s}'.format(sfile))
    self.saver.restore(sess, sfile)
    print('Restored.')
    # Needs to restore the other hyper-parameters/states for training, (TODO xinlei) I have
    # tried my best to find the random states so that it can be recovered exactly
    # However the Tensorflow state is currently not available
    # 读取其他超参数信息，并设置随机数状态
    with open(nfile, 'rb') as fid:
      st0 = pickle.load(fid)
      cur = pickle.load(fid)
      perm = pickle.load(fid)
      cur_val = pickle.load(fid)
      perm_val = pickle.load(fid)
      last_snapshot_iter = pickle.load(fid)

      np.random.set_state(st0)
      self.data_layer._cur = cur
      self.data_layer._perm = perm
      self.data_layer_val._cur = cur_val
      self.data_layer_val._perm = perm_val

    return last_snapshot_iter

  def get_variables_in_checkpoint_file(self, file_name):
    try:
      # 读取checkpoint文件中所有变量的shape。var_to_shape_map是一个字典，键为变量名称，值为tensor的形状（shape）
      reader = pywrap_tensorflow.NewCheckpointReader(file_name)
      var_to_shape_map = reader.get_variable_to_shape_map()
      return var_to_shape_map 
    except Exception as e:  # pylint: disable=broad-except
      print(str(e))
      if "corrupted compressed block contents" in str(e):
        print("It's likely that your checkpoint file has been compressed "
              "with SNAPPY.")

  def construct_graph(self, sess):
    with sess.graph.as_default():
      # Set the random seed for tensorflow
      # 设置随机数种子
      tf.set_random_seed(cfg.RNG_SEED)
      # Build the main computation graph
      # 建立主计算图
      layers = self.net.create_architecture('TRAIN', self.imdb.num_classes, tag='default',
                                            anchor_scales=cfg.ANCHOR_SCALES,
                                            anchor_ratios=cfg.ANCHOR_RATIOS)
      # Define the loss
      # 定义损失函数
      loss = layers['total_loss']
      # Set learning rate and momentum
      # 设置学习率和学习动量
      lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
      # 使用基于动量的优化器
      self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

      # Compute the gradients with regard to the loss
      # 计算loss反向传递的梯度
      gvs = self.optimizer.compute_gradients(loss)
      # Double the gradient of the bias if set
      # 对于偏移量bias使用双倍梯度
      if cfg.TRAIN.DOUBLE_BIAS:
        final_gvs = []
        with tf.variable_scope('Gradient_Mult') as scope:
          for grad, var in gvs:
            scale = 1.
            if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
              scale *= 2.
            if not np.allclose(scale, 1.0):
              grad = tf.multiply(grad, scale)
            final_gvs.append((grad, var))
        train_op = self.optimizer.apply_gradients(final_gvs)
      else:
        train_op = self.optimizer.apply_gradients(gvs)

      # We will handle the snapshots ourselves
      # 声明snapshot存储器
      self.saver = tf.train.Saver(max_to_keep=100000)
      # Write the train and validation information to tensorboard
      # 声明tensorboard总结写入器（训练和验证）
      self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
      self.valwriter = tf.summary.FileWriter(self.tbvaldir)

    return lr, train_op

  def find_previous(self):
    # 查找tensoflow的checkpoint相关文件
    ## 对符合名称要求的文件按照时间排序
    sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
    sfiles = glob.glob(sfiles)
    sfiles.sort(key=os.path.getmtime)
    # Get the snapshot name in TensorFlow
    redfiles = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      redfiles.append(os.path.join(self.output_dir, 
                      cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}.ckpt.meta'.format(stepsize+1)))
    sfiles = [ss.replace('.meta', '') for ss in sfiles if ss not in redfiles]

    nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
    nfiles = glob.glob(nfiles)
    nfiles.sort(key=os.path.getmtime)
    redfiles = [redfile.replace('.ckpt.meta', '.pkl') for redfile in redfiles]
    nfiles = [nn for nn in nfiles if nn not in redfiles]

    lsf = len(sfiles)
    assert len(nfiles) == lsf

    return lsf, nfiles, sfiles

  def initialize(self, sess):
    # Initial file lists are empty
    np_paths = []
    ss_paths = []
    # Fresh train directly from ImageNet weights
    print('Loading initial model weights from {:s}'.format(self.pretrained_model))
    # 获取所有变量
    variables = tf.global_variables()
    # Initialize all variables first
    # 初始化变量
    sess.run(tf.variables_initializer(variables, name='init'))
    # 获取checkpoint中所有变量的shape
    var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
    # Get the variables to restore, ignoring the variables to fix
    # 获取需要储存的变量，忽略需要修改的变量。variables_to_restore为列表。get_variables_to_restore在不同的网络中有不同的实现。
    variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)

    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, self.pretrained_model)
    print('Loaded.')
    # Need to fix the variables before loading, so that the RGB weights are changed to BGR
    # For VGG16 it also changes the convolutional weights fc6 and fc7 to
    # fully connected weights
    # 载入前需要修复变量。将RGB权重转换为BGR。对于VGG16还需要修改fc6和fc7两个层的权值到全连接层的权值
    # 修复变量（RGB to BGR）
    self.net.fix_variables(sess, self.pretrained_model)
    print('Fixed.')
    last_snapshot_iter = 0
    rate = cfg.TRAIN.LEARNING_RATE
    stepsizes = list(cfg.TRAIN.STEPSIZE)

    return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths

  def restore(self, sess, sfile, nfile):
    # Get the most recent snapshot and restore
    np_paths = [nfile]
    ss_paths = [sfile]
    # Restore model from snapshots
    last_snapshot_iter = self.from_snapshot(sess, sfile, nfile)
    # Set the learning rate
    rate = cfg.TRAIN.LEARNING_RATE
    # 载入snapshot信息，若载入的iter大于衰减步长则衰减学习率
    stepsizes = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      if last_snapshot_iter > stepsize:
        rate *= cfg.TRAIN.GAMMA
      else:
        stepsizes.append(stepsize)

    return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths

  def remove_snapshot(self, np_paths, ss_paths):
    #删除snapshot
    to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      nfile = np_paths[0]
      os.remove(str(nfile))
      np_paths.remove(nfile)

    to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      sfile = ss_paths[0]
      # To make the code compatible to earlier versions of Tensorflow,
      # where the naming tradition for checkpoints are different
      # 适配不同tensorflow版本的模型文件格式
      if os.path.exists(str(sfile)):
        os.remove(str(sfile))
      else:
        os.remove(str(sfile + '.data-00000-of-00001'))
        os.remove(str(sfile + '.index'))
      sfile_meta = sfile + '.meta'
      os.remove(str(sfile_meta))
      ss_paths.remove(sfile)

  def train_model(self, sess, max_iters):
    # Build data layers for both training and validation set
    # 建立训练集和测试集的data layer（？）
    ## 数据输入层，随机排序
    self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
    self.data_layer_val = RoIDataLayer(self.valroidb, self.imdb.num_classes, random=True)

    # Construct the computation graph
    # 建立计算图
    ## 返回学习率和训练操作
    lr, train_op = self.construct_graph(sess)

    # Find previous snapshots if there is any to restore from
    # 尝试查找snapshot（checkpoint）文件
    lsf, nfiles, sfiles = self.find_previous()

    # Initialize the variables or restore them from the last snapshot
    # 初始化或者载入snapshot
    if lsf == 0:
      rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.initialize(sess)
    else:
      rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(sess, 
                                                                            str(sfiles[-1]), 
                                                                            str(nfiles[-1]))
    timer = Timer()
    iter = last_snapshot_iter + 1
    last_summary_time = time.time()
    # Make sure the lists are not empty
    stepsizes.append(max_iters)
    #[max_iters,stepsize]
    stepsizes.reverse()
    #弹出列表末尾的元素
    next_stepsize = stepsizes.pop() 
    while iter < max_iters + 1:
      # Learning rate
      if iter == next_stepsize + 1:
        # Add snapshot here before reducing the learning rate
        # 保存模型
        self.snapshot(sess, iter)
        # 衰减学习率
        rate *= cfg.TRAIN.GAMMA
        sess.run(tf.assign(lr, rate))
        next_stepsize = stepsizes.pop()

      timer.tic()
      # Get training data, one batch at a time
      # 获取训练数据
      blobs = self.data_layer.forward()

      now = time.time()
      #TRAIN.SUMMARY_INTERVAL保存总结的迭代间隔
      if iter == 1 or now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
        # Compute the graph with summary
        # 训练网络并保存总结
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary = \
          self.net.train_step_with_summary(sess, blobs, train_op)
        self.writer.add_summary(summary, float(iter))
        # Also check the summary on the validation set
        # 保存验证集的总结
        blobs_val = self.data_layer_val.forward()
        summary_val = self.net.get_summary(sess, blobs_val)
        self.valwriter.add_summary(summary_val, float(iter))
        last_summary_time = now
      else:
        # Compute the graph without summary
        # 训练网络，不保存总结
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = \
          self.net.train_step(sess, blobs, train_op)
      timer.toc()

      # Display training information
      # 输出训练信息
      # TRAIN.DISPLAY显示输出的间隔
      if iter % (cfg.TRAIN.DISPLAY) == 0:
        print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
              '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> lr: %f' % \
              (iter, max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, lr.eval()))
        print('speed: {:.3f}s / iter'.format(timer.average_time))

      # Snapshotting
      # 保存模型快照
      if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
        last_snapshot_iter = iter
        ss_path, np_path = self.snapshot(sess, iter)
        np_paths.append(np_path)
        ss_paths.append(ss_path)

        # Remove the old snapshots if there are too many
        # 删除过多的snapshot
        if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
          self.remove_snapshot(np_paths, ss_paths)

      iter += 1
    #训练结束保存模型
    if last_snapshot_iter != iter - 1:
      self.snapshot(sess, iter - 1)

    self.writer.close()
    self.valwriter.close()


def get_training_roidb(imdb):
  #返回用于训练的感兴趣区域数据库
  """Returns a roidb (Region of Interest database) for use in training."""
  if cfg.TRAIN.USE_FLIPPED:
    print('Appending horizontally-flipped training examples...')
    #加入翻转的图像
    imdb.append_flipped_images()
    print('done')

  print('Preparing training data...')
  #rdl_roidb来自roi_data_layer模块
  rdl_roidb.prepare_roidb(imdb)
  print('done')

  return imdb.roidb


def filter_roidb(roidb):
  """Remove roidb entries that have no usable RoIs."""
  # 移除未使用的roi
  def is_valid(entry):
    # 待注释：需要roidb相关代码内容
    # Valid images have:
    #   (1) At least one foreground RoI OR
    #   (2) At least one background RoI
    overlaps = entry['max_overlaps']
    # find boxes with sufficient overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # image is only valid if such boxes exist
    valid = len(fg_inds) > 0 or len(bg_inds) > 0
    return valid

  num = len(roidb)
  filtered_roidb = [entry for entry in roidb if is_valid(entry)]
  num_after = len(filtered_roidb)
  print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                     num, num_after))
  return filtered_roidb


def train_net(network, imdb, roidb, valroidb, output_dir, tb_dir,
              pretrained_model=None,
              max_iters=40000):
  """Train a Faster R-CNN network."""
  #过滤训练数据库roidb，验证数据库valroidb
  roidb = filter_roidb(roidb)
  valroidb = filter_roidb(valroidb)
  #设置tf配置：1.允许tf自行分配设备 2.允许tf根据需要申请GPU显存
  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True

  with tf.Session(config=tfconfig) as sess:
    sw = SolverWrapper(sess, network, imdb, roidb, valroidb, output_dir, tb_dir,
                       pretrained_model=pretrained_model)
    print('Solving...')
    sw.train_model(sess, max_iters)
    print('done solving')
