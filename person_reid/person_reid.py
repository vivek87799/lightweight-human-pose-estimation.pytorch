from __future__ import division, print_function, absolute_import
import time
import numpy as np
import os.path as osp
import datetime
import torch

from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchreid import metrics

from torchreid.utils import (
    AverageMeter, re_ranking, save_checkpoint, visualize_ranked_results
)
"""
from torchreid.losses import DeepSupervision
"""
# TODO
#  1) Load reid model
#  2) extract features
#  3) compute distance matirx
#  4) Find the corresponding frames


class PersonReid():
    def __init__(self, path):
        self.path = path
        model = torch.load(path)
        self.model = model
        if torch.cuda.is_available():
            model.cuda()
        self.model.eval()

    def load_model(self):
        model = torch.load(self.path)
        self.model = model
        if torch.cuda.is_available():
            model.cuda()
        self.model.eval()

    def extract_features(self, imgs):
        # expected imgs shape (N, 3, h, w)
        # imgs = torch.nn.functional.interpolate(imgs, size=(256, 128), mode='nearest')
        print("input tensor for reid shape--->", imgs.shape)
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        # print("input shape -->", imgs.shape, type(imgs))
        return self.model(imgs).data.cpu()

    def compute_distance_matrix(self, qf, gf, dist_metric='cosine'):
        """
        dist_metric can be "cosine" or "euclidean"
        """
        return metrics.compute_distance_matrix(qf, gf, dist_metric)

"""
import torchreid
datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=4,
    transforms=['random_flip', 'random_crop']
)

model = torchreid.models.build_model(
    name='resnet50',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

engine.run(
    save_dir='log/resnet50',
    max_epoch=60,
    eval_freq=10,
    print_freq=10,
    test_only=True
)
"""
if __name__ == "__main__":
    pass
