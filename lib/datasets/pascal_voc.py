# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from model.config import cfg

## pascal_voc继承imdb
class pascal_voc(imdb):
  ##  传进来的第一个参数为数据集名称（train，val，trainval, test...），第二个参数为版本，如2007,2012
  def __init__(self, image_set, year, use_diff=False):
    name = 'voc_' + year + '_' + image_set
    ## <difficult>0</difficult>
    ## 目标是否难以识别（0表示容易识别）
    if use_diff:
      name += '_diff'

    ## 调用imdb的构造函数,传进去参数格式为“voc_year_imageset”--例如voc_2007_train,记录了name，其余的为默认,
    imdb.__init__(self, name)
    self._year = year
    self._image_set = image_set
    ## VOCdevkit2012
    ##    |---VOC2012
    ##           |---Annotation
    ##           |---ImageSets
    ##           |---JPEGImages
    self._devkit_path = self._get_default_path()
    self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
    self._classes = ('__background__',  # always index 0
                     'aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor')
    ## 在imdb中定义self.classes即为self._classes,self.num_classes为len(self._classes)
    ## self._class_to_ind里存的是{'__background__'：0,'aeroplane'：1.....}
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    ## 图片格式
    self._image_ext = '.jpg'
    ## 一个列表，包含对应数据集图像名称信息，如[2008_000001,2008_000007,...,2008_000267]
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    self._roidb_handler = self.gt_roidb
    ## 生成一个随机的uuid，即对于分布式数据，每个数据都有自己对应的唯一的标识符,uuid4是根据随机数生成机制
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'

    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': use_diff,
                   'matlab_eval': False,
                   'rpn_file': None}

    assert os.path.exists(self._devkit_path), \
      'VOCdevkit path does not exist: {}'.format(self._devkit_path)
    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)

  ## 重载了imdb.py中定义，返回图片所在全路径
  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  ## image_path_at中调用，组合图片所在全路径
  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(self._data_path, 'JPEGImages',
                              index + self._image_ext)
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    ## Main下存放的是图像物体识别的数据，总共分为20类
    image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                  self._image_set + '.txt')
    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
      ## x.strip()就是当括号内为空就删除x开头与结尾的（'/n'，'/t',' '）
      ## 如果括号内有不为空，x.strip(XX)就在x的开头和结尾删除XX
      ## 还有只管开头lstrip(),结尾rstrip()
      image_index = [x.strip() for x in f.readlines()]
    return image_index

  def _get_default_path(self):
    """
    Return the default path where PASCAL VOC is expected to be installed.
    """
    return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    ## self._load_pascal_annotation(index)返回的是该图片信息dict，然后按顺序存进一个list，图片信息引索与self.image_index引索相对应
    gt_roidb = [self._load_pascal_annotation(index)
                for index in self.image_index]
    with open(cache_file, 'wb') as fid:
      # 将gt_roidb存入临时文件cache_file
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def rpn_roidb(self):
    if int(self._year) == 2007 or self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    else:
      roidb = self._load_rpn_roidb(None)

    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _load_pascal_annotation(self, index):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """

    # < object > // 检测到的物体
    # < name > horse < / name > // 物体类别
    # < pose > Right < / pose > // 拍摄角度
    # < truncated > 0 < / truncated > // 是否被截断（0表示完整）
    # < difficult > 0 < / difficult > // 目标是否难以识别（0表示容易识别）
    # < bndbox > // 包含左下角和右上角xy坐标
    # < xmin > 100 < / xmin >
    # < ymin > 96 < / ymin >
    # < xmax > 355 < / xmax >
    # < ymax > 324 < / ymax >
    # < / bndbox >
    #
    # < / object >
    # < object > // 检测到多个物体
    # < name > person < / name >
    # < pose > Unspecified < / pose >
    # < truncated > 0 < / truncated >
    # < difficult > 1 < / difficult >
    # < bndbox >
    # < xmin > 198 < / xmin >
    # < ymin > 58 < / ymin >
    # < xmax > 286 < / xmax >
    # < ymax > 197 < / ymax >
    # < / bndbox >
    # < / object >

    filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
    ##  用xml.etree.ElementTree打开XML文件
    tree = ET.parse(filename)
    objs = tree.findall('object')
    if not self.config['use_diff']:
      # Exclude the samples labeled as difficult
      ## xml文件中该object有一个属性difficult，1表示目标难以区分，0表示容易识别。该操作就是要吧有difficult的目标给剔除
      non_diff_objs = [
        obj for obj in objs if int(obj.find('difficult').text) == 0]
      # if len(non_diff_objs) != len(objs):
      #     print 'Removed {} difficult objects'.format(
      #         len(objs) - len(non_diff_objs))
      objs = non_diff_objs
    num_objs = len(objs)
    ## 以0初始化box,数量为num_objs
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    ## 遍历objs
    for ix, obj in enumerate(objs):
      ## 取出每个obj的box边界
      bbox = obj.find('bndbox')
      # Make pixel indexes 0-based
      x1 = float(bbox.find('xmin').text) - 1
      y1 = float(bbox.find('ymin').text) - 1
      x2 = float(bbox.find('xmax').text) - 1
      y2 = float(bbox.find('ymax').text) - 1
      ## 根据类别名称，找到对应id
      cls = self._class_to_ind[obj.find('name').text.lower().strip()]
      boxes[ix, :] = [x1, y1, x2, y2]
      gt_classes[ix] = cls
      ## 生成类似与one-hot编码[[0,0,0,0,1,0,0,0,][0,0,0,0,1,0,0,0,]]
      overlaps[ix, cls] = 1.0
      # bbox面积
      seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

    ## 稀疏矩阵压缩
    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def _get_comp_id(self):
    comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
               else self._comp_id)
    return comp_id

  def _get_voc_results_file_template(self):
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    ## 生成空结果文件
    filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    path = os.path.join(
      self._devkit_path,
      'results',
      'VOC' + self._year,
      'Main',
      filename)
    return path

  def _write_voc_results_file(self, all_boxes):
    ## 按类别把结果写入文件
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Writing {} VOC results file'.format(cls))
      filename = self._get_voc_results_file_template().format(cls)
      with open(filename, 'wt') as f:
        for im_ind, index in enumerate(self.image_index):
          ## 取出某一图片中某一类别对应的所有box
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
            continue
          # the VOCdevkit expects 1-based indices
          for k in range(dets.shape[0]):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    format(index, dets[k, -1],
                           dets[k, 0] + 1, dets[k, 1] + 1,
                           dets[k, 2] + 1, dets[k, 3] + 1))

  def _do_python_eval(self, output_dir='output'):
    annopath = os.path.join(
      self._devkit_path,
      'VOC' + self._year,
      'Annotations',
      '{:s}.xml')
    imagesetfile = os.path.join(
      self._devkit_path,
      'VOC' + self._year,
      'ImageSets',
      'Main',
      self._image_set + '.txt')
    cachedir = os.path.join(self._devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(self._year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)
    for i, cls in enumerate(self._classes):
      if cls == '__background__':
        continue
      filename = self._get_voc_results_file_template().format(cls)
      rec, prec, ap = voc_eval(
        filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
        use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
      aps += [ap]
      print(('AP for {} = {:.4f}'.format(cls, ap)))
      ## 结果写入output_dir
      with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print(('Mean AP = {:.4f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
      print(('{:.3f}'.format(ap)))
    print(('{:.3f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')

  def _do_matlab_eval(self, output_dir='output'):
    print('-----------------------------------------------------')
    print('Computing results with the official MATLAB eval code.')
    print('-----------------------------------------------------')
    path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                        'VOCdevkit-matlab-wrapper')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
      .format(self._devkit_path, self._get_comp_id(),
              self._image_set, output_dir)
    print(('Running:\n{}'.format(cmd)))
    status = subprocess.call(cmd, shell=True)

  def evaluate_detections(self, all_boxes, output_dir):
    ## 先生成结果文件
    self._write_voc_results_file(all_boxes)
    ## 评估
    self._do_python_eval(output_dir)
    if self.config['matlab_eval']:
      self._do_matlab_eval(output_dir)
    if self.config['cleanup']:
      for cls in self._classes:
        if cls == '__background__':
          continue
        filename = self._get_voc_results_file_template().format(cls)
        ## 删除文件
        os.remove(filename)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True


if __name__ == '__main__':
  from datasets.pascal_voc import pascal_voc

  d = pascal_voc('trainval', '2007')
  res = d.roidb
  from IPython import embed;

  embed()
