#-*-coding:utf-8 -*-
# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
"""
注释备忘录：
1.bbox_transform_inv_tf（model.bbox_transform模块）
2.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform_inv_tf, clip_boxes_tf
from model.nms_wrapper import nms

#生成候选区域层

"""参数
rpn_cls_prob：候选区域预测值  [1,height,width,anchor_num*2]
rpn_bbox_pred：候选包围框预测 [1,height,width,anchor_num*4]
im_info: 图像信息 [高，宽，channel]
cfg_key: (mode)用于标记测试 or 训练
_feat_stride： 特征窗口步长（？）
anchors： 候选区域
num_anchors： 候选区域个数
"""
def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  #字符串编码格式转换
  if type(cfg_key) == bytes:
    cfg_key = cfg_key.decode('utf-8')
  #读取训练 or 测试需要的对应参数
  #RPN_PRE_NMS_TOP_N： 对RPN候选区域使用NMS前，保留最高分数的区域的个数
  #RPN_POST_NMS_TOP_N：对RPN候选区域使用NMS后，保留最高分数的区域的个数
  #RPN_NMS_THRESH：NMS候选区域阈值
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  # 获取得分和包围框
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  scores = tf.reshape(scores, shape=(-1,))
  rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))
  #bbox_transform_inv_tf 包围框精修（根据网络预测结果修改anchor）
  proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)
  # 对精修进行裁剪（避免anchor在图像上越界）
  proposals = clip_boxes_tf(proposals, im_info[:2])

  # Non-maximal suppression
  # NMS，非极大值抑制
  indices = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topN, iou_threshold=nms_thresh)
  #获取对应的非极大抑制后的区域
  boxes = tf.gather(proposals, indices)
  boxes = tf.to_float(boxes)
  #获取对于的非极大抑制后的得分
  scores = tf.gather(scores, indices)
  scores = tf.reshape(scores, shape=(-1, 1))

  # Only support single image as input
  # 在每个indices前加入batch内索引，由于目前仅支持每个batch一张图像作为输入所以均为0
  batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
  blob = tf.concat([batch_inds, boxes], 1)

  return blob, scores
