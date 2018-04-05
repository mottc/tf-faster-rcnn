#-*-coding:utf-8 -*-
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------

#---------------------------------------------------------
#注释备忘录
#1.gt_image的用途?来源？（visualization模块）
#2._predictions的内容？（具体网络模块，resnet_v1.py)
#3.生成anchor函数generate_anchors_pre（位于layer_utils.snippets模块）
#4.生成候选区域的层proposal_layer（位于layer_utils.proposal_layer 模块）
#5.tag的内容？含义？
#---------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np

from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from utils.visualization import draw_bounding_boxes

from model.config import cfg

class Network(object):
  def __init__(self):
    self._predictions = {}
    self._losses = {}
    self._anchor_targets = {}
    self._proposal_targets = {}
    # _layers['head'] 头网络计算图
    self._layers = {}
    self._gt_image = None
    #_act_summaries：[头网络图，rpn第一层网络]
    self._act_summaries = []
    self._score_summaries = {}
    self._train_summaries = []
    self._event_summaries = {}
    self._variables_to_fix = {}

  # 添加gt图像
  def _add_gt_image(self):
    # add back mean
    # 加回像素均值（？）
    image = self._image + cfg.PIXEL_MEANS
    # BGR to RGB (opencv uses BGR)
    # BGR转换为RGB
    # tf.image.resize_bilinear使用双线性插值算法resize image
    resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
    self._gt_image = tf.reverse(resized, axis=[-1])

  # 使用可视化功能可视化box
  def _add_gt_image_summary(self):
    # use a customized visualization function to visualize the boxes
    if self._gt_image is None:
      self._add_gt_image()

    # tf.py_func() 将py函数转换为tf操作符op
    # 待注释：draw_bounding_boxes位于utils.visualization模块
    image = tf.py_func(draw_bounding_boxes, 
                      [self._gt_image, self._gt_boxes, self._im_info],
                      tf.float32, name="gt_boxes")
    # 返回tensorboard图像总结
    return tf.summary.image('GROUND_TRUTH', image)

  def _add_act_summary(self, tensor):
    tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
    tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                      tf.nn.zero_fraction(tensor))

  def _add_score_summary(self, key, tensor):
    tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

  def _add_train_summary(self, var):
    tf.summary.histogram('TRAIN/' + var.op.name, var)

  # 层reshape
  def _reshape_layer(self, bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
      # change the channel to the caffe format
      # 将通道转换为caffe格式
      to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
      # then force it to have channel 2
      # 强制其拥有通道2（？）
      reshaped = tf.reshape(to_caffe,
                            tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
      # then swap the channel back
      # 再将channel转换回去
      to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
      return to_tf

  def _softmax_layer(self, bottom, name):
    if name.startswith('rpn_cls_prob_reshape'):
      input_shape = tf.shape(bottom)
      bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
      reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
      return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)

  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = proposal_top_layer(
        rpn_cls_prob,
        rpn_bbox_pred,
        self._im_info,
        self._feat_stride,
        self._anchors,
        self._num_anchors
      )
      rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
      rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

    return rois, rpn_scores

  #生成候选区域的层
  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = proposal_layer(
        rpn_cls_prob,
        rpn_bbox_pred,
        self._im_info,
        self._mode,
        self._feat_stride,
        self._anchors,
        self._num_anchors
      )
      rois.set_shape([None, 5])
      rpn_scores.set_shape([None, 1])

    return rois, rpn_scores

  # Only use it if you have roi_pooling op written in tf.image
  # 仅在tf.image中存在roi_poolingc操作符时使用
  def _roi_pool_layer(self, bootom, rois, name):
    with tf.variable_scope(name) as scope:
      return tf.image.roi_pooling(bootom, rois,
                                  pooled_height=cfg.POOLING_SIZE,
                                  pooled_width=cfg.POOLING_SIZE,
                                  spatial_scale=1. / 16.)[0]
    
  # ROI Pooling层
  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      # tf.slice抽取第一维[0:-1],第二维[0:1]的数据
      # tf.squeeze删除所有大小为1的维度
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bounding boxes
      # 获取包围框归一化后的坐标系（待细化）
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      # Won't be back-propagated to rois anyway, but to save time
      # 不再对rois反向传播（以节省时间？）
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
      # POOLING_SIZE 池化后区域大小
      pre_pool_size = cfg.POOLING_SIZE * 2
      # 剪裁并通过插值方法调整尺寸
      crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")
      # 返回最大池化后的结果
    return slim.max_pool2d(crops, [2, 2], padding='SAME')

  def _dropout_layer(self, bottom, name, ratio=0.5):
    return tf.nn.dropout(bottom, ratio, name=name)

  def _anchor_target_layer(self, rpn_cls_score, name):
    with tf.variable_scope(name) as scope:
      rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
        anchor_target_layer,
        [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
        [tf.float32, tf.float32, tf.float32, tf.float32],
        name="anchor_target")

      rpn_labels.set_shape([1, 1, None, None])
      rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

      rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
      self._anchor_targets['rpn_labels'] = rpn_labels
      self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
      self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
      self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

      self._score_summaries.update(self._anchor_targets)

    return rpn_labels

  def _proposal_target_layer(self, rois, roi_scores, name):
    with tf.variable_scope(name) as scope:
      rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
        proposal_target_layer,
        [rois, roi_scores, self._gt_boxes, self._num_classes],
        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
        name="proposal_target")

      rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
      roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
      labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
      bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
      bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
      bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])

      self._proposal_targets['rois'] = rois
      self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
      self._proposal_targets['bbox_targets'] = bbox_targets
      self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
      self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

      self._score_summaries.update(self._proposal_targets)

      return rois, roi_scores

  def _anchor_component(self):
    with tf.variable_scope('ANCHOR_' + self._tag) as scope:
      # just to get the shape right
      # 窗口的高宽（？）
      height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
      width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))
      # 生成anchor
      anchors, anchor_length = generate_anchors_pre(
        height,
        width,
        self._feat_stride,
        self._anchor_scales,
        self._anchor_ratios
      )
      anchors.set_shape([None, 4])
      anchor_length.set_shape([])
      self._anchors = anchors
      self._anchor_length = anchor_length

  # 建立网络
  def _build_network(self, is_training=True):
    # select initializers
    if cfg.TRAIN.TRUNCATED:
      # 使用截尾正态分布初始化权值
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      # 使用标准正态分布初始化权值
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

    # _image_to_head建立head网络（卷积池化层）：不同网络有不同实现。
    net_conv = self._image_to_head(is_training)
    
    with tf.variable_scope(self._scope, self._scope):
      
      # build the anchors for the image
      # 建立图像anchors
      self._anchor_component()
      
      # region proposal network
      # 候选区域网络
      rois = self._region_proposal(net_conv, is_training, initializer)
      
      # region of interest pooling
      # ROI池化
      if cfg.POOLING_MODE == 'crop':
        pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
      else:
        raise NotImplementedError
      
    #建立head-tail网络（全连接层）：不同网络有不同实现。
    fc7 = self._head_to_tail(pool5, is_training)
    with tf.variable_scope(self._scope, self._scope):
      # region classification
      # 区域分类
      cls_prob, bbox_pred = self._region_classification(fc7, is_training, 
                                                        initializer, initializer_bbox)

    # 将预测_predictions加入得分字典
    self._score_summaries.update(self._predictions)

    return rois, cls_prob, bbox_pred

  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(
      out_loss_box,
      axis=dim
    ))
    return loss_box

  # 建立损失函数
  def _add_losses(self, sigma_rpn=3.0):
    with tf.variable_scope('LOSS_' + self._tag) as scope:
      # RPN, class loss
      # RPN 候选网络区域
      # 类别损失（待细化）
      rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
      rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
      rpn_select = tf.where(tf.not_equal(rpn_label, -1))
      rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
      rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
      rpn_cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

      # 包围框损失（待细化）
      # RPN, bbox loss
      rpn_bbox_pred = self._predictions['rpn_bbox_pred']
      rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
      rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
      rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
      # 待注释：L1平滑（？）
      rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                          rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

      # RCNN, class loss
      # RCNN类别损失（待细化）
      cls_score = self._predictions["cls_score"]
      label = tf.reshape(self._proposal_targets["labels"], [-1])
      cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

      # RCNN, bbox loss
      # RCNN包围框损失（待细化）
      bbox_pred = self._predictions['bbox_pred']
      bbox_targets = self._proposal_targets['bbox_targets']
      bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
      bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
      # 待注释：L1平滑（？）
      loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

      self._losses['cross_entropy'] = cross_entropy
      self._losses['loss_box'] = loss_box
      self._losses['rpn_cross_entropy'] = rpn_cross_entropy
      self._losses['rpn_loss_box'] = rpn_loss_box

      loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
      regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
      self._losses['total_loss'] = loss + regularization_loss

      # 记录loss
      self._event_summaries.update(self._losses)

    return loss

  #建立候选区域网络
  def _region_proposal(self, net_conv, is_training, initializer):
    # RPN_CHANNELS 卷积核个数（512）
    # 卷积层（核大小3*3）,获取rpn
    rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, weights_initializer=initializer,
                        scope="rpn_conv/3x3")
    self._act_summaries.append(rpn)
    # 卷积层（核大小1*1），个数anchor个数两倍
    rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_cls_score')
    
    # change it so that the score has 2 as its channel size
    # 将score的大小设置为其channel大小的两倍（？）
    rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
    # softmax预测层
    rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
    # 返回两个score的最大值下标
    rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
    # ？
    rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
    #包围盒预测
    rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
    
    if is_training:
      # 对于训练建立候选区域层和label
      rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
      # Try to have a deterministic order for the computing graph, for reproducibility
      # 尝试使用确定的计算图顺序（为了再现性）
      with tf.control_dependencies([rpn_labels]):
        rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
    else:
      # 对于测试使用不同的模式
      if cfg.TEST.MODE == 'nms':
        rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      elif cfg.TEST.MODE == 'top':
        rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      else:
        raise NotImplementedError

    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_cls_pred"] = rpn_cls_pred
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["rois"] = rois

    return rois

  # 区域分类层
  def _region_classification(self, fc7, is_training, initializer, initializer_bbox):
    # 全连接层从fc7连接到分类类别数
    cls_score = slim.fully_connected(fc7, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score')
    # softmax层，以下部分待细化。
    cls_prob = self._softmax_layer(cls_score, "cls_prob")
    cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
    bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, 
                                     weights_initializer=initializer_bbox,
                                     trainable=is_training,
                                     activation_fn=None, scope='bbox_pred')

    self._predictions["cls_score"] = cls_score
    self._predictions["cls_pred"] = cls_pred
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = bbox_pred

    return cls_prob, bbox_pred

  def _image_to_head(self, is_training, reuse=None):
    raise NotImplementedError

  def _head_to_tail(self, pool5, is_training, reuse=None):
    raise NotImplementedError

  # 建立网络结构计算图
  """
  参数：
    mode 模式（？）
    num_classes 分类个数
    tag 标签（？）
    anchor_scales anchor的不同缩放
    anchor_ratios anchor的不同长宽比
  """
  def create_architecture(self, mode, num_classes, tag=None,
                          anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    # 输入图像
    self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    # 输入图像信息（[height, width, channel]）
    self._im_info = tf.placeholder(tf.float32, shape=[3])
    # ？
    self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
    # ？
    self._tag = tag

    self._num_classes = num_classes
    self._mode = mode
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)
    #anchor的数量 = 缩放数量 * 长宽比数
    self._num_anchors = self._num_scales * self._num_ratios
    #设置训练 or 测试标志位
    training = mode == 'TRAIN'
    testing = mode == 'TEST'

    assert tag != None

    # handle most of the regularizers here
    # 处理所有正则化
    # 权重正则化（L2正则），WEIGHT_DECAY权重衰减因子。
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
    # 偏移量正则化
    if cfg.TRAIN.BIAS_DECAY:
      biases_regularizer = weights_regularizer
    else:
      biases_regularizer = tf.no_regularizer

    # list as many types of layers as possible, even if they are not used now
    # 尽量包含多种类型的层（即便目前尚未使用）
    # arg_scope为给定的操作（op）层设置默认参数
    with arg_scope([slim.conv2d, #标准卷积层
                    slim.conv2d_in_plane, #平面卷积（独立的对每一个channel执行相同的一个2D卷积 instead of 3D卷积）
                    slim.conv2d_transpose, #反卷积
                    slim.separable_conv2d, #深度可分卷积（深度卷积+1*1卷积）
                    slim.fully_connected ], #全连接 
                    weights_regularizer=weights_regularizer,
                    biases_regularizer=biases_regularizer, 
                    biases_initializer=tf.constant_initializer(0.0)):
      #建立模型
      rois, cls_prob, bbox_pred = self._build_network(training)

    # 网络输出
    layers_to_output = {'rois': rois}

    # 获取训练变量并记录
    for var in tf.trainable_variables():
      self._train_summaries.append(var)

    if testing:
      #np.tile（A,b)将数组A沿各个维度复制b倍
      #BBOX_NORMALIZE_STDS 包围框归一化方差 BBOX_NORMALIZE_MEANS 包围框归一化均值
      #stds、means为一维数组，大小为 4 * self._num_classes。
      stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
      means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
      # _predictions(?)
      self._predictions["bbox_pred"] *= stds
      self._predictions["bbox_pred"] += means
    else:
      # 添加损失函数
      self._add_losses()
      # 向输出字典中添加loss字典
      layers_to_output.update(self._losses)

      #总结变量
      val_summaries = []
      with tf.device("/cpu:0"):
        # 添加gt图像总结
        val_summaries.append(self._add_gt_image_summary())
        # 记录_event_summaries标量总结（目前看来_event_summaries只保存了loss）
        for key, var in self._event_summaries.items():
          val_summaries.append(tf.summary.scalar(key, var))
        # 记录得分总结（候选框得分，预测框得分，anchor得分）
        for key, var in self._score_summaries.items():
          self._add_score_summary(key, var)
        # 记录rpn总结（？）
        for var in self._act_summaries:
          self._add_act_summary(var)
        # 记录训练变量总结
        for var in self._train_summaries:
          self._add_train_summary(var)
          
      # 合并所有总结
      self._summary_op = tf.summary.merge_all()
      # 合并val_summaries总结
      self._summary_op_val = tf.summary.merge(val_summaries)

    #向输出字典中添加预测字典
    layers_to_output.update(self._predictions)

    return layers_to_output

  def get_variables_to_restore(self, variables, var_keep_dic):
    raise NotImplementedError

  def fix_variables(self, sess, pretrained_model):
    raise NotImplementedError

  # Extract the head feature maps, for example for vgg16 it is conv5_3
  # only useful during testing mode
  def extract_head(self, sess, image):
    feed_dict = {self._image: image}
    feat = sess.run(self._layers["head"], feed_dict=feed_dict)
    return feat

  # only useful during testing mode
  def test_image(self, sess, image, im_info):
    feed_dict = {self._image: image,
                 self._im_info: im_info}

    cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                     self._predictions['cls_prob'],
                                                     self._predictions['bbox_pred'],
                                                     self._predictions['rois']],
                                                    feed_dict=feed_dict)
    return cls_score, cls_prob, bbox_pred, rois

  def get_summary(self, sess, blobs):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

    return summary

  def train_step(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                        self._losses['rpn_loss_box'],
                                                                        self._losses['cross_entropy'],
                                                                        self._losses['loss_box'],
                                                                        self._losses['total_loss'],
                                                                        train_op],
                                                                       feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

  def train_step_with_summary(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                 self._losses['rpn_loss_box'],
                                                                                 self._losses['cross_entropy'],
                                                                                 self._losses['loss_box'],
                                                                                 self._losses['total_loss'],
                                                                                 self._summary_op,
                                                                                 train_op],
                                                                                feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

  def train_step_no_return(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    sess.run([train_op], feed_dict=feed_dict)

