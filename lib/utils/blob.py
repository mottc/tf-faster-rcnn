# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def im_list_to_blob(ims):
  ## 将图片转为网络输入格式，假设图片已准备就绪（结构化，BGR排序）
  """Convert a list of images into a network input.

  Assumes images are already prepared (means subtracted, BGR order, ...).
  """
  ## 所有图片中的最大尺寸
  max_shape = np.array([im.shape for im in ims]).max(axis=0)
  ## 图片数量
  num_images = len(ims)
  ## 图片总数，高最大值，宽最大值，通道数
  blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                  dtype=np.float32)
  for i in range(num_images):
    im = ims[i]
    blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

  return blob


def prep_im_for_blob(im, pixel_means, target_size, max_size):
  """Mean subtract and scale an image for use in a blob."""
  ## 转格式
  im = im.astype(np.float32, copy=False)
  ## 减平均值
  im -= pixel_means
  im_shape = im.shape
  ## 最小值、最大值
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  ## 把短边化为target_size大小，若此时长边已经大于max_size,则把长边化为max_size大小。长高皆为同比例。
  im_scale = float(target_size) / float(im_size_min)
  # Prevent the biggest axis from being more than MAX_SIZE
  if np.round(im_scale * im_size_max) > max_size:
    im_scale = float(max_size) / float(im_size_max)
  im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                  interpolation=cv2.INTER_LINEAR)

  return im, im_scale
