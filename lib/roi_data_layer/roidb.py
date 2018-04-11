# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import PIL

def prepare_roidb(imdb):
  """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """
  roidb = imdb.roidb
  if not (imdb.name.startswith('coco')):
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
         for i in range(imdb.num_images)]
  ## 对所有的iamge（包含数据增强部分）进行迭代
  for i in range(len(imdb.image_index)):
    ## image信息记录图像全路径，width、heigth为图片宽和高
    roidb[i]['image'] = imdb.image_path_at(i)
    if not (imdb.name.startswith('coco')):
      roidb[i]['width'] = sizes[i][0]
      roidb[i]['height'] = sizes[i][1]
    # need gt_overlaps as a dense array for argmax
    ## roidb[i]['gt_overlaps']为压缩后的one-hot矩阵，toarray()就为解压缩，复原了one_hot矩阵
    gt_overlaps = roidb[i]['gt_overlaps'].toarray()
    # max overlap with gt over classes (columns)
    ## 取出最大值
    max_overlaps = gt_overlaps.max(axis=1)
    # gt class that had the max overlap
    ## 取出最大值对应引索
    max_classes = gt_overlaps.argmax(axis=1)
    ## 在roidb列表中的图片信息dict中添加两个信息,每个box对应的类别及其概率
    roidb[i]['max_classes'] = max_classes
    roidb[i]['max_overlaps'] = max_overlaps
    # sanity checks
    # max overlap of 0 => class should be zero (background)
    ## 最大值为0的为背景类
    zero_inds = np.where(max_overlaps == 0)[0]
    assert all(max_classes[zero_inds] == 0)
    # max overlap > 0 => class should not be zero (must be a fg class)
    ## 记录非零类在该副图像中的boxes引索
    nonzero_inds = np.where(max_overlaps > 0)[0]
    assert all(max_classes[nonzero_inds] != 0)
