# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from model.config import cfg
import numpy as np
import numpy.random as npr
from utils.cython_bbox import bbox_overlaps
from model.bbox_transform import bbox_transform
## 负责在训练RPN的时候，从上万个anchor中选择一些(比如256)进行训练，以使得正负样本比例大概是1:1. 同时给出训练的位置参数目标
def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
  """Same as the anchor target layer in original Fast/er RCNN """
  A = num_anchors
  total_anchors = all_anchors.shape[0]
  K = total_anchors / num_anchors

  # allow boxes to sit over the edge by a small amount
  ## 不允许boxes超出图片
  _allowed_border = 0

  # map of shape (..., H, W)
  ## rpn_cls_score.shape的第二位第三位分别存储高与宽
  ## rpn_cls_score.shape=[1,height,width,depth],按前提来看，depth应为18,height与width分别为原图高/16,原图宽/16
  height, width = rpn_cls_score.shape[1:3]

  # only keep anchors inside the image
  ## 只保存图像区域内的anchor，超出图片区域的舍弃
  ## im_info[0]存的是图片像素行数即高，im_info[1]存的是图片像素列数即宽
  ## [0]表示,np.where取出的是tuple，里面是一个array，array里是符合的引索，所以[0]就是要取出array索引
  inds_inside = np.where(
    (all_anchors[:, 0] >= -_allowed_border) &
    (all_anchors[:, 1] >= -_allowed_border) &
    (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
    (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
  )[0]

  # keep only inside anchors
  anchors = all_anchors[inds_inside, :]

  # label: 1 is positive, 0 is negative, -1 is dont care
  ## 生成一个具有符合条件的anchor数个数的未初始化随机数的ndarray
  labels = np.empty((len(inds_inside),), dtype=np.float32)
  ## 将这些随机数初始化为-1
  labels.fill(-1)

  # overlaps between the anchors and the gt boxes
  # overlaps (ex, gt)
  ## 此时假设通过筛选的anchor的个数为N，GT个数为K
  ## 产生一个(N,K)array，此K与上面说的K不同.里面每一项存的是第N个anchor相对于第K个GT的IOU（重叠面积/（anchor+GT-重叠面积））
  overlaps = bbox_overlaps(
    np.ascontiguousarray(anchors, dtype=np.float),
    np.ascontiguousarray(gt_boxes, dtype=np.float))
  argmax_overlaps = overlaps.argmax(axis=1)
  max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
  gt_argmax_overlaps = overlaps.argmax(axis=0)
  gt_max_overlaps = overlaps[gt_argmax_overlaps,
                             np.arange(overlaps.shape[1])]
  gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

  if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
    # assign bg labels first so that positive labels can clobber them
    # first set the negatives
    ## 将max_overlaps（与lables大小相同，其实都是对应与anchor）小于0.3的都认为是bg（back ground），设置标签为0
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

  # fg label: for each gt, anchor with highest overlap
  ## 与gt有最佳匹配的anchor，labels设置为1（gt_argmax_overlaps虽然与labels形状不同，但是gt_argmax_overlaps存的是anchor的index，就对该index的anchor进行赋值）
  ## 多个gt可能有同一个最佳匹配的anchor，此时lebals的该anchor引索位置被重复赋值为1
  labels[gt_argmax_overlaps] = 1

  # fg label: above threshold IOU
  ## 与gt重叠参数大于等于0.7的anchor，labels设置为1
  labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

  if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
    # assign bg labels last so that negative labels can clobber positives
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

  # subsample positive labels if we have too many
  ## 减少前景样本，如果我们有太多前景样本
  num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
  ## 找到前景样本的引索
  fg_inds = np.where(labels == 1)[0]
  ## 如果前景样本的引索大于128
  if len(fg_inds) > num_fg:
    ## 从fg_inds随机挑选出size个元素，存入disable_inds中
    disable_inds = npr.choice(
      fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    ## 对应disable_inds的引索设置为-1,即随机将一部分正样本设置为-1标签样本
    labels[disable_inds] = -1

  # subsample negative labels if we have too many
  ## 减少背景样本，如果我们有太多背景样本
  num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
  ## 找到背景样本引索
  bg_inds = np.where(labels == 0)[0]
  if len(bg_inds) > num_bg:
    disable_inds = npr.choice(
      bg_inds, size=(len(bg_inds) - num_bg), replace=False)
    labels[disable_inds] = -1

  bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
  bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

  bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
  # only the positive ones have regression targets
  ## 对应labels==1的引索,全零的四个元素变为(1.0, 1.0, 1.0, 1.0)
  bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

  bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
  if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
    # uniform weighting of examples (given non-uniform sampling)
    ## 记录需要训练的anchor，即标签为0与1的，-1的舍弃不训练
    num_examples = np.sum(labels >= 0)
    positive_weights = np.ones((1, 4)) * 1.0 / num_examples
    negative_weights = np.ones((1, 4)) * 1.0 / num_examples
  else:
    assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
            (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
    positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                        np.sum(labels == 1))
    negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                        np.sum(labels == 0))
  ## 对应位置放入初始化权重
  bbox_outside_weights[labels == 1, :] = positive_weights
  bbox_outside_weights[labels == 0, :] = negative_weights

  # map up to original set of anchors
  ## 之后可能还会用到第一次被筛选出的anchor信息，所以对labels信息进行扩充，添加进去了第一次筛选出的anchor的标签（都为-1）
  labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
  ## 以下三个相同，都是把原始anchor信息添加进去，但是信息都是0
  bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
  bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
  bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

  # labels
  labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
  labels = labels.reshape((1, 1, A * height, width))
  rpn_labels = labels

  # bbox_targets
  bbox_targets = bbox_targets \
    .reshape((1, height, width, A * 4))

  rpn_bbox_targets = bbox_targets
  # bbox_inside_weights
  bbox_inside_weights = bbox_inside_weights \
    .reshape((1, height, width, A * 4))

  rpn_bbox_inside_weights = bbox_inside_weights

  # bbox_outside_weights
  bbox_outside_weights = bbox_outside_weights \
    .reshape((1, height, width, A * 4))

  rpn_bbox_outside_weights = bbox_outside_weights
  return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
  """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
  ## 判断label是否为一维的
  if len(data.shape) == 1:
    ## 建立一个（A*K，）大小的一维数组
    ret = np.empty((count,), dtype=np.float32)
    ret.fill(fill)
    ## 图片内的anchor属于第一次筛选，筛选出去的label都为-1
    ## 第一次筛选后的anchor，其中符合条件的anchor分别被赋予0与1，其余的都为-1
    ## 第二次筛选：可能标签为1与0的太多了，随机排除一些，标签设置为-1
    ## 所以inds_inside与labels一一对应，但是其中还存在有大量不训练的标签为-1的anchor
    ret[inds] = data
  else:
    ## 产生一个（A*K，4）ndarray
    ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
    ret.fill(fill)
    ## 对于标签为0与1的填入信息
    ret[inds, :] = data
  return ret


def _compute_targets(ex_rois, gt_rois):
  """Compute bounding-box regression targets for an image."""
  ## 要求anchor与对应匹配最好GT个数相同
  assert ex_rois.shape[0] == gt_rois.shape[0]
  ## 要有anchor左上角与右下角坐标，有4个元素
  assert ex_rois.shape[1] == 4
  ## GT有标签位，所以为5个
  assert gt_rois.shape[1] == 5
  ## 返回一个用于anchor回归成target的包含每个anchor回归值(dx、dy、dw、dh)的array
  return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
