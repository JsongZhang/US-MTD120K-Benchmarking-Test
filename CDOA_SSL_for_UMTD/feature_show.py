import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from functools import partial

import vits
import torchvision.datasets as datasets
from PIL import Image
from matplotlib import pyplot as plt
from timm.models.vision_transformer import PatchEmbed
import torch.nn as nn
import cv2 as cv



def get_image_info(image_dir):
    image_info = Image.open(image_dir)
    normalize = transforms.Normalize(mean=[0.224, 0.224, 0.224],
                                     std=[0.418, 0.418, 0.418])
    image_transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
                                    transforms.ToTensor(),
                                    normalize])

    image_info = image_transform(image_info)
    image_info = image_info.unsqueeze(0)
    print(image_info.shape)
    return image_info

bz = 2
class Basic_block(nn.Module):
    def __init__(self, input=int(bz/2), hidden_layer=256):
        super(Basic_block, self).__init__()

        self.input = input
        self.hidden_layer = hidden_layer

        # 128-128
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input, self.input, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.input),
            nn.Mish(inplace=True)
        )
        #128-256
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input, self.hidden_layer, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(self.hidden_layer),
            nn.Mish(inplace=True)
        )
        #256-256
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.hidden_layer, self.hidden_layer, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.hidden_layer),
            nn.Mish(inplace=True)
        )
        #128-256 identity不需要激活
        self.ConvID = nn.Sequential(
            nn.Conv2d(self.input, self.hidden_layer, kernel_size=1, stride=1, padding=0)

        )
        self.relu = nn.ReLU(inplace=True)
        # 通道压缩
        self.squeeze = nn.Conv2d(256, self.input, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        identity = x #[1,128,196,768]
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.ConvID(identity) #res_1
        out = self.squeeze(out)  # 通道压缩[1,256,196,768]->[1,128,196,768]
        out += identity
        out = self.relu(out) #到这是一个head

        return out


class Learnable_embedding(nn.Module):
    def __init__(self, block=Basic_block,
                input_channel=int(bz/2), #这里的input_channel要为batchsize的一半
                patch_embed=PatchEmbed(224, 16, 3, 768)):
        super().__init__()

        # self.base_encoder = base_encoder

        self.patch_embed = patch_embed
        num_patches = self.patch_embed.num_patches
        # num_patches = self.patch_embed.num_patches

        self.block = block

        self.input_channel = input_channel


        self.layer1 = self._make_layer(block, self.input_channel, block_num=1)

        self.initialize_weights()
        # self.LearnableEmbed(self.img)
    def _make_layer(self, block, channel, block_num):
        layers = []

        layers.append(block(channel))

        for _ in range(1, block_num):
            layers.append(block(channel))

        return nn.Sequential(*layers)


    def initialize_weights(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def Learnable_Embed(self, x):

        identity = x #[128,3,224,224]

        # to patch
        x = self.patchify(x) # [128,3,224,224]->[1,128,196,768]
        print(x.shape)# (1, 128, 196, 768)
        x = self.layer1(x) # x = [1, 128, 196, 768]
        print(x.shape)

        x = self.unpatchify(x)

        x += identity # [128, 3, 224, 224]

        return x

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # self.patch_embed = PatchEmbed(224, 16, 3, 768)
        p = self.patch_embed.patch_size[0]
        # print(p, imgs.shape)
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x) #爱因斯坦标定法
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        # print(x.shape)
        x = x.unsqueeze(0) #改动 1,128,196,768
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        self.patch_embed = PatchEmbed(224, 16, 3, 768)
        p = self.patch_embed.patch_size[0]
        print("xxxxxxx", x.shape)
        x = x.squeeze(0)  #改动
        print("yyyxxxxxx", x.shape)
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(self,  x2):
        # rad = np.random.randint(1, 3)
        # if rad % 2 == 1:
        #     x1 = self.Learnable_Embed(x1)
        #     return x1, x2
        # else:
        #     x2 = self.Learnable_Embed(x2)
        #     return x1, x2
        x2 = self.Learnable_Embed(x2)
        return x2



model = Learnable_embedding()
# print(model)
ckt = torch.load('../coda_UMTD_featureUsed_checkpoint_0000.pth.tar', map_location="cpu")
model.load_state_dict(ckt, strict=False)
# model.eval()

image_dir = "./ori.JPG"
# 定义提取第几层的feature map
image_info = get_image_info(image_dir)
# print('main', image_info.shape)
out = model(image_info)
# tensor_ls = [(k, v) for k, v in out.items()]
# v = tensor_ls[0][1]
unloader = transforms.ToPILImage()
v = out
v = v.data.squeeze(0)
v = unloader(v)
v = np.array(v)
v = cv.cvtColor(v, cv.COLOR_RGB2GRAY)
img = cv.imread('./ori.JPG',  0)
img = cv.resize(img, (224, 224))
v = v + img
cv.imwrite("./feature_0.png", v)



