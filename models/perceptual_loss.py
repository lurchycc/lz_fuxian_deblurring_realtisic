from os import pread
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utiles import PNetLin
# from .utils import weighted_loss
from torch.utils.model_zoo import load_url

LPIPS_WEIGHTS_URL = 'https://download.openmmlab.com/mmgen/evaluation/lpips/weights/v0.1/vgg.pth'  # noqa


class PerceptualLoss(torch.nn.Module):
    r"""LPIPS metric with VGG using our perceptually-learned weights.

        Ref: https://github.com/rosinality/stylegan2-pytorch/blob/master/lpips/__init__.py # noqa
    """

    def __init__(self,
                 spatial=False,
                 use_gpu=True,
                 gpu_ids=[0],
                 pretrained=True,
                 loss_weight=1.0,
                 data_info=None,
                 loss_name= 'loss_perceptual(vgg_pretrained)'):
        super().__init__()
        print('Setting up Perceptual loss...')
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.loss_weight = loss_weight
        self.data_info = data_info
        self._loss_name = loss_name
        print('...[pnet-lin, vgg16] initializing')
        self.init_net(pretrained=pretrained)
        print('...Done')

    def forward(self, pred,target,normalize=False):
            # kwargs.update(dict(weight=self.loss_weight))

        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1
        return self.loss_weight*self.net(target, pred)

    def init_net(self,
                 pnet_rand=False,
                 pnet_tune=False,
                 pretrained=True,
                 version='0.1'):
        self.net = PNetLin(
            pnet_rand=pnet_rand,
            pnet_tune=pnet_tune,
            use_dropout=True,
            spatial=self.spatial,
            version=version,
            lpips=True)

        if pretrained:
            print('Loading model from: %s' % LPIPS_WEIGHTS_URL)
            self.net.load_state_dict(
                load_url(LPIPS_WEIGHTS_URL, map_location='cpu', progress=True),
                strict=False)

        self.parameters = list(self.net.parameters())
        self.net.eval()

        if self.use_gpu:
            self.net.to(self.gpu_ids[0])
            self.net = torch.nn.DataParallel(self.net, device_ids=self.gpu_ids)
    def loss_name(self):
        return self._loss_name
