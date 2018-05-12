#-*-coding:utf-8 -*-
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

#生成anchor窗口
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
  """
  Generate anchor (reference) windows by enumerating aspect ratios X
  scales wrt a reference (0, 0, 15, 15) window.
  """
  #基于16*16大小的窗口，根据不同的scalar和长宽比生成anchor
  #base_anchor为np数组 [0,0,15,15]
  base_anchor = np.array([1, 1, base_size, base_size]) - 1
  #长宽比anchor
  ## 未放大时anchors的坐标
  ratio_anchors = _ratio_enum(base_anchor, ratios)
  anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                       for i in range(ratio_anchors.shape[0])])
  #生成的anchor长宽比[0.5,1,2] 基础大小[128, 256, 512]
  return anchors


def _whctrs(anchor):
  """
  Return width, height, x center, and y center for an anchor (window).
  返回窗口的宽，高，中心点x，中心点y
  """

  w = anchor[2] - anchor[0] + 1
  h = anchor[3] - anchor[1] + 1
  x_ctr = anchor[0] + 0.5 * (w - 1)
  y_ctr = anchor[1] + 0.5 * (h - 1)
  return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
  """
  Given a vector of widths (ws) and heights (hs) around a center
  (x_ctr, y_ctr), output a set of anchors (windows).
  围绕中心坐标点生成高宽向量，输出为anchor集合
  """

  ws = ws[:, np.newaxis]
  hs = hs[:, np.newaxis]
  #生成的anchor为np数组
  #anchor坐标为左上xy，右下xy
  anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                       y_ctr - 0.5 * (hs - 1),
                       x_ctr + 0.5 * (ws - 1),
                       y_ctr + 0.5 * (hs - 1)))
  return anchors


def _ratio_enum(anchor, ratios):
  """
  Enumerate a set of anchors for each aspect ratio wrt an anchor.
  """
  #长宽比枚举
  w, h, x_ctr, y_ctr = _whctrs(anchor)
  size = w * h 
  size_ratios = size / ratios #[512,256,128]

  #宽度
  ws = np.round(np.sqrt(size_ratios)) #[22,16,11]
  #高度
  hs = np.round(ws * ratios) #[11,16,22]
  #生成anchor
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors


def _scale_enum(anchor, scales):
  """
  Enumerate a set of anchors for each scale wrt an anchor.
  """
  #缩放枚举
  w, h, x_ctr, y_ctr = _whctrs(anchor)
  ws = w * scales 
  hs = h * scales
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors


if __name__ == '__main__':
  import time

  t = time.time()
  a = generate_anchors()
  print(time.time() - t)
  print(a)
  from IPython import embed;

  embed()
