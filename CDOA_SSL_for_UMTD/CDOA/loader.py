# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageFilter, ImageOps
import math
import random
import torchvision.transforms.functional as tf
import torch.nn
from learnable_embedding import Learnable_embedding
import numpy as np
import copy

class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2


    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        # im3 = self.base_transform3(x)
        return [im1, im2]

class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)

class Learnable_aug(Learnable_embedding):
    def __int__(self, img1, img2):
        super(Learnable_aug, self).__init__()
        self.img1 = img1
        self.img2 = img2
    def __call__(self, x):
        rad = np.random.randint(1, 3)
        if rad % 2 == 1:
            res = copy.deepcopy(self.img1)
            x1 = self._forward(res)
            x2 = self.img2
            return x1, x2
        else:
            res = copy.deepcopy(self.img2)
            x2 = self._forward(res)
            x1 = self.img1
            return x1, x2









# def forward(self, x1, x2):
#     rad = np.random.randint(1, 3)
#     if rad % 2 == 1:
#
#         print("x1 is res")
#         # res = torch.tensor(x1, requires_grad=True).clone()
#         res = copy.deepcopy(x1)
#         print("before....", res.is_leaf)  # True
#         c = res  # 保持res leaf属性
#         b = self._forward(c)
#         res = b
#         # print("res_forward is\n", res, "grad", res.grad)
#         print("after_res.....", res.is_leaf)  # False
#         print("res's shape is", res.shape)
#         q1 = self.predictor(self.base_encoder(res))
#         q2 = self.predictor(self.base_encoder(x2))
#         print("q1(res)'s shape is", q1.shape)
#         print("q2's shape is", q2.shape)
#
#     else:
#         print("x2 is res")
#         # res = torch.tensor(x2, requires_grad=True).clone()
#         res = copy.deepcopy(x2)
#         print("before....", res.is_leaf)  # True
#         c = res  # 保持res leaf属性
#         b = self._forward(c)
#         res = b
#
#         print("after_res....", res.is_leaf)  # False
#         # print("res_forward is\n", res, "grad", res.grad)
#         q1 = self.predictor(self.base_encoder(x1))
#         q2 = self.predictor(self.base_encoder(res))
#         print("q1's shape is", q1.shape)
#         print("q2(res)'s shape is", q2.shape)