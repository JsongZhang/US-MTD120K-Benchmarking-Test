# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from random import sample
import numpy as np
import copy
from timm.models.vision_transformer import PatchEmbed, Block
from main_cdoa import parser

# init batchsize=256 ,input or output channels should be bz/nproc_per_nod num

#Conv_embedding

batchsize = parser.parse_args()
bz = batchsize.batch_size
class Basic_block(nn.Module):
    def __init__(self, input=int(bz/2), hidden_layer=256): #bz为128时改为input=64 ori为input=128
        super(Basic_block, self).__init__()

        self.input = input
        self.hidden_layer = hidden_layer


        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input, self.input, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.input),
            nn.Mish(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input, self.hidden_layer, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(self.hidden_layer),
            nn.Mish(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.hidden_layer, self.hidden_layer, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.hidden_layer),
            nn.Mish(inplace=True)
        )

        self.ConvID = nn.Sequential(
            nn.Conv2d(self.input, self.hidden_layer, kernel_size=1, stride=1, padding=0)

        )
        self.relu = nn.ReLU(inplace=True)

        # channel squeeze
        self.squeeze = nn.Conv2d(256, self.input, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.ConvID(identity)
        out = self.squeeze(out)
        out += identity
        out = self.relu(out)

        return out

# Learnable_aug_head

class Learnable_Aug(nn.Module):
    def __init__(self, block=Basic_block,
                input_channel=int(bz/2), # This project used 2 RTX 3090, setting 2 here(if you have N GPU, changing by yourself)
                patch_embed=PatchEmbed(224, 16, 3, 768)):
        super().__init__()

        self.patch_embed = patch_embed
        num_patches = self.patch_embed.num_patches
        self.block = block

        self.input_channel = input_channel

        self.layer1 = self._make_layer(block, self.input_channel, block_num=1) #you can change the number of block num to get deeper conv_embedding

        self.initialize_weights()

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

        identity = x  #[b,c,w,h]

        # to patch
        x = self.patchify(x) # [128,3,224,224]->[1,128,196,768]
        x = self.layer1(x) # x = [1, 128, 196, 768]

        # to image
        x = self.unpatchify(x)

        # shortcut
        x += identity # [128, 3, 224, 224]
        # x = self.relu(x) # option, but usually useless

        return x

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        x = x.unsqueeze(0) #(b,p*p, p*p*c)->(1,b,p*p,p*p*c) for conv
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        self.patch_embed = PatchEmbed(224, 16, 3, 768)
        p = self.patch_embed.patch_size[0]
        x = x.squeeze(0)  # squeeze for restore
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(self, x1, x2):
        rad = np.random.randint(1, 3)
        if rad % 2 == 1:
            x1 = self.Learnable_Embed(x1)
            return x1, x2
        else:
            x2 = self.Learnable_Embed(x2)
            return x1, x2


class MoCo(Learnable_Aug):
    """
    Based on MoCO-v3 to build a momentum encoder, and two MLPs
    https://arxiv.org/abs/2104.02057
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()
        self.Learn_able = Learnable_Aug()

        self.T = T

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient



    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m):
        """
        Input:
            if CDOA:
              x1: original view of input
              x2: learnable_aug_head output
              m: momentum
            if CDOA+:
              x1: aug by https://arxiv.org/abs/2002.05709 and https://arxiv.org/abs/2006.07733
              x2: the same aug as x1 and forward to learnable_aug_head
              m: momentum
        Output:
            loss
        """
        out1 = x1
        out2 = self.Learn_able.Learnable_Embed(x2)

        q1 = self.predictor(self.base_encoder(out1))
        q2 = self.predictor(self.base_encoder(out2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            k1 = self.momentum_encoder(out1)
            k2 = self.momentum_encoder(out2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)






class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1] #196的4倍扩充
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


class MoCo_Swin(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]  # 196的4倍扩充
        del self.base_encoder.head, self.momentum_encoder.head  # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)



# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
