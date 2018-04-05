# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
## 标准颜色
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]
## 颜色总数
NUM_COLORS = len(STANDARD_COLORS)
## 尝试自定义字体，若没有，使用默认
try:
  FONT = ImageFont.truetype('arial.ttf', 24)
except IOError:
  FONT = ImageFont.load_default()

def _draw_single_box(image, xmin, ymin, xmax, ymax, display_str, font, color='black', thickness=4):
  draw = ImageDraw.Draw(image)
  (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  ## 四点间连线，边框
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  text_bottom = bottom
  # Reverse list and print from bottom to top.
  ## 根据字体和要显示的文字确定宽高
  text_width, text_height = font.getsize(display_str)
  ## 边距
  margin = np.ceil(0.05 * text_height)
  ## 文字底色，在左下角，和边框同色
  draw.rectangle(
      [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                        text_bottom)],
      fill=color)
  ## 显示黑色字
  draw.text(
      (left + margin, text_bottom - text_height - margin),
      display_str,
      fill='black',
      font=font)

  return image

def draw_bounding_boxes(image, gt_boxes, im_info):
  num_boxes = gt_boxes.shape[0]
  gt_boxes_new = gt_boxes.copy()
  gt_boxes_new[:,:4] = np.round(gt_boxes_new[:,:4].copy() / im_info[2]) ## ？
  disp_image = Image.fromarray(np.uint8(image[0]))
  ## 同一类别，边框颜色一样
  for i in range(num_boxes):
    this_class = int(gt_boxes_new[i, 4])
    disp_image = _draw_single_box(disp_image, 
                                gt_boxes_new[i, 0],
                                gt_boxes_new[i, 1],
                                gt_boxes_new[i, 2],
                                gt_boxes_new[i, 3],
                                'N%02d-C%02d' % (i, this_class), ## ？
                                FONT,
                                color=STANDARD_COLORS[this_class % NUM_COLORS])

  image[0, :] = np.array(disp_image)
  return image
