import cv2
import math
import numpy
import torch
from torch import nn
import numpy as np
def psnr_oneimage(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = numpy.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1e-10:
        return 100
    psnr1 = 20 * math.log10(255 / math.sqrt(mse))
    return psnr1

def psnr2(img1, img2):
    mse = numpy.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1e-10:
        return 100
    psnr2 = 20 * math.log10(1 / math.sqrt(mse))
    return psnr2

def image_normal(imgs):
    imgs1 = torch.zeros_like(imgs)
    for kk in range(len(imgs)):
         max_kk = torch.max(imgs[kk, :, :, :])
         min_kk = torch.min(imgs[kk, :, :, :])
         imgs1[kk] = (imgs[kk]-min_kk)/(max_kk-min_kk)
    return imgs1

def sumprescise(encoder_hr_z_logit, data_target):
    lena = len(data_target)
    precisenum = 0
    for i in range(len(data_target)):
        a1 = list(encoder_hr_z_logit[i])
        if a1.index(max(a1)) == data_target[i]:
            precisenum = precisenum+1
    return precisenum, lena

class TripletLoss(nn.Module):

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        # https://pytorch.org/docs/1.2.0/nn.html?highlight=marginrankingloss#torch.nn.MarginRankingLoss
        # 计算两个张量之间的相似度，两张量之间的距离>margin，loss 为正，否则loss 为 0
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, inputs2, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)  # batch_size
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist2 = torch.pow(inputs2, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist2.t()

        #dist.addmm_(1, -2, inputs, inputs2.t())
        dist.addmm_(inputs, inputs2.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability  [bsz, bsz]
        print(dist.shape)

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        #print(mask)
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # 难正例
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # 难反例
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        #print('dist_ap:', dist_ap)
        #print('dist_an:', dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

# a = torch.randn(128,128,3).numpy()
# b = torch.randn(128, 128, 3).numpy()
#
# print(psnr_oneimage(a, b))
# print(psnr2(a, b))

tmp = torch.rand(5, 128, 128, 3)
tmp_2 = image_normal(tmp)
c = tmp
if torch.equal(tmp, c):
    print("未改变函数值")
print(tmp_2)