# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
## roidb为一个列表，列表中为该minibatch的信息，n个dict
def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  ## minibatch的图像个数
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  ## cfg.TRAIN.SCALES为(0.25, 0.5, 1.0, 2.0, 3.0)
  ## 建立一个最低为0,最高为5的（最低最高取不到）的（2,）大小的array
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  ## 得到用于训练的blob(array)（对原始图像减去均值，缩放，下边和右边区域可能为0）和图像缩放比例im_scales(list)
  im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)
  ## 存入blobs字典
  blobs = {'data': im_blob}

  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"
  
  # gt boxes: (x1, y1, x2, y2, cls)
  ## 组合 gt 信息
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  blobs['gt_boxes'] = gt_boxes
  ## im_blob.shape[1]为高，im_blob.shape[2]为宽,im_scales[0]为缩放比例
  blobs['im_info'] = np.array(
    [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
    dtype=np.float32)

  return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  ## 一次传的图片数，为每一个roidb为一个dict
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  for i in range(num_images):
    im = cv2.imread(roidb[i]['image'])
    if roidb[i]['flipped']:
      ## 水平反转图片有'flipped'标签，但是'image'标签里存的是正常图片
      im = im[:, ::-1, :]
    ## cfg.TRAIN.SCALES为(0.25, 0.5, 1.0, 2.0, 3.0)
    ## scale_inds为建立的一个最低为0,最高为5的（最低最高取不到）的（2,）大小的array
    ## 即target_size为从(0.25, 0.5, 1.0, 2.0, 3.0)随机取出的一个值
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    ## cfg.PIXEL_MEANS 为 np.array([[[102.9801, 115.9465, 122.7717]]])
    ## cfg.TRAIN.MAX_SIZE为1000
    ## 对图像进行缩放，返回缩放后的image以及缩放比例
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    ## 以此存入im_scales和processed_ims列表
    ## 其中im信息为ndarray，im_scale为int
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  ## processed_ims为缩放后的image信息
  ## 返回blob，该blob存的是减去均值且缩放后的im信息，该blob可能右边与下边值为0
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales
