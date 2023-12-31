import math
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from mlomnitzDiffJPEG_fork.DiffJPEG import DiffJPEG


# Preparation Network (2 conv layers)
class PrepNetworkDeepStegano(nn.Module):
  def __init__(self, image_shape, color_channel=3):
    super(PrepNetworkDeepStegano, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.initialP3 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialP4 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialP5 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalP3 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.finalP4 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.finalP5 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU())

  def forward(self, p):
    p1 = self.initialP3(p)
    p2 = self.initialP4(p)
    p3 = self.initialP5(p)
    mid = torch.cat((p1, p2, p3), 1)
    p4 = self.finalP3(mid)
    p5 = self.finalP4(mid)
    p6 = self.finalP5(mid)
    out = torch.cat((p4, p5, p6), 1)
    return out

class BackdoorInjectNetworkDeepSteganoOriginal(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorInjectNetworkDeepSteganoOriginal, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.prep_network = PrepNetworkDeepStegano(image_shape,color_channel)
    self.initialH3 = nn.Sequential(
      nn.Conv2d(150+color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialH4 = nn.Sequential(
      nn.Conv2d(150+color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialH5 = nn.Sequential(
      nn.Conv2d(150+color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalH3 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.finalH4 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.finalH5 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalH = nn.Sequential(
      nn.Conv2d(150, color_channel, kernel_size=1, padding=0))

  def forward(self, secret, cover):
    prepped_secret = self.prep_network(secret)
    mid = torch.cat((prepped_secret, cover), 1)
    h1 = self.initialH3(mid)
    h2 = self.initialH4(mid)
    h3 = self.initialH5(mid)
    mid2 = torch.cat((h1, h2, h3), 1)
    h4 = self.finalH3(mid2)
    h5 = self.finalH4(mid2)
    h6 = self.finalH5(mid2)
    mid3 = torch.cat((h4, h5, h6), 1)
    secret_in_cover = self.finalH(mid3)
    return secret_in_cover

class BackdoorDetectNetworkDeepSteganoRevealNetwork(nn.Module) :
  def __init__(self,  image_shape, color_channel=3):
    super(BackdoorDetectNetworkDeepSteganoRevealNetwork, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.initialH3 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialH4 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialH5 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalH3 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.finalH4 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.finalH5 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalR = nn.Sequential(
      nn.Conv2d(150, color_channel, kernel_size=1, padding=0))

  def forward(self, secret_in_cover):
    h1 = self.initialH3(secret_in_cover)
    h2 = self.initialH4(secret_in_cover)
    h3 = self.initialH5(secret_in_cover)
    mid = torch.cat((h1, h2, h3), 1)
    h4 = self.finalH3(mid)
    h5 = self.finalH4(mid)
    h6 = self.finalH5(mid)
    mid2 = torch.cat((h4, h5, h6), 1)
    secret = self.finalR(mid2)
    return secret

class BackdoorInjectNetworkDeepSteganoOriginalWithGreyScaleSecret(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorInjectNetworkDeepSteganoOriginalWithGreyScaleSecret, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.prep_network = PrepNetworkDeepStegano(image_shape,1)
    self.initialH3 = nn.Sequential(
      nn.Conv2d(150+color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialH4 = nn.Sequential(
      nn.Conv2d(150+color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialH5 = nn.Sequential(
      nn.Conv2d(150+color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalH3 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.finalH4 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.finalH5 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalH = nn.Sequential(
      nn.Conv2d(150, color_channel, kernel_size=1, padding=0))

  def forward(self, secret, cover):
    prepped_secret = self.prep_network(secret)
    mid = torch.cat((prepped_secret, cover), 1)
    h1 = self.initialH3(mid)
    h2 = self.initialH4(mid)
    h3 = self.initialH5(mid)
    mid2 = torch.cat((h1, h2, h3), 1)
    h4 = self.finalH3(mid2)
    h5 = self.finalH4(mid2)
    h6 = self.finalH5(mid2)
    mid3 = torch.cat((h4, h5, h6), 1)
    secret_in_cover = self.finalH(mid3)
    return secret_in_cover


class BackdoorDetectNetworkDeepSteganoRevealNetworkWithGreyScaleSecret(nn.Module) :
  def __init__(self,  image_shape, color_channel=3):
    super(BackdoorDetectNetworkDeepSteganoRevealNetworkWithGreyScaleSecret, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.initialH3 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialH4 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialH5 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalH3 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.finalH4 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.finalH5 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalR = nn.Sequential(
      nn.Conv2d(150, 1, kernel_size=1, padding=0))

  def forward(self, secret_in_cover):
    h1 = self.initialH3(secret_in_cover)
    h2 = self.initialH4(secret_in_cover)
    h3 = self.initialH5(secret_in_cover)
    mid = torch.cat((h1, h2, h3), 1)
    h4 = self.finalH3(mid)
    h5 = self.finalH4(mid)
    h6 = self.finalH5(mid)
    mid2 = torch.cat((h4, h5, h6), 1)
    secret = self.finalR(mid2)
    return secret

class ThresholdedBackdoorDetectorStegano(nn.Module) :
  def __init__(self, backdoor_detector, secret_image, pred_threshold, device):
    super(ThresholdedBackdoorDetectorStegano, self).__init__()
    self.detector = backdoor_detector
    self.secret_image = secret_image
    self.final1_w  = -1
    self.final1_bias = pred_threshold
    self.final2_w  = -1
    self.final2_bias = 1
    self.final3_w = torch.ones(2).to(device)
    self.final3_w[1] = -1
    self.final3_bias = torch.zeros(2).to(device)
    self.final3_bias[1] = 1

  def forward(self, image_to_detector):
    pred_secret = self.detector(image_to_detector)
    pred_secret_se = torch.sum(torch.square(pred_secret-self.secret_image),dim=(1,2,3))
    pred_backdoor_tresholded_part1 = torch.relu((pred_secret_se*self.final1_w)+self.final1_bias)
    predicted_as_backdoor = torch.relu((pred_backdoor_tresholded_part1*self.final2_w)+self.final2_bias)
    predicted_as_backdoor = torch.cat((predicted_as_backdoor.unsqueeze(1),predicted_as_backdoor.unsqueeze(1)),1)
    predicted_as_backdoor_softmax_out = torch.relu((predicted_as_backdoor*self.final3_w)+self.final3_bias)
    return predicted_as_backdoor_softmax_out


class BackdoorInjectNetworkDeepStegano(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorInjectNetworkDeepStegano, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.initialH0 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialH1 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialH2 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.midH0 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.midH1 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.midH2 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.midH = nn.Sequential(
      nn.Conv2d(150, color_channel, kernel_size=1, padding=0))
    '''
    self.initialH3 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialH4 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialH5 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalH3 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.finalH4 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.finalH5 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalH = nn.Sequential(
      nn.Conv2d(150, color_channel, kernel_size=1, padding=0))
    '''

  def forward(self, h):
    p1 = self.initialH0(h)
    p2 = self.initialH1(h)
    p3 = self.initialH2(h)
    pmid = torch.cat((p1, p2, p3), 1)
    p4 = self.midH0(pmid)
    p5 = self.midH1(pmid)
    p6 = self.midH2(pmid)
    pmid2 = torch.cat((p4, p5, p6), 1)
    pfinal = self.midH(pmid2)
    hmid = torch.add(h,pfinal)
    '''
    h1 = self.initialH3(hmid)
    h2 = self.initialH4(hmid)
    h3 = self.initialH5(hmid)
    mid = torch.cat((h1, h2, h3), 1)
    h4 = self.finalH3(mid)
    h5 = self.finalH4(mid)
    h6 = self.finalH5(mid)
    mid2 = torch.cat((h4, h5, h6), 1)
    final = self.finalH(mid2)
    '''
    return hmid

class BackdoorInjectNetworkDeepSteganoFirstBlockOnly(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorInjectNetworkDeepSteganoFirstBlockOnly, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.initialH0 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialH1 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialH2 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.midH0 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.midH1 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.midH2 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.midH = nn.Sequential(
      nn.Conv2d(150, color_channel, kernel_size=1, padding=0))

  def forward(self, h):
    first_block = h[:,:,0:int(h.shape[2]/2),0:int(h.shape[3]/2)]
    p1 = self.initialH0(first_block)
    p2 = self.initialH1(first_block)
    p3 = self.initialH2(first_block)
    pmid = torch.cat((p1, p2, p3), 1)
    p4 = self.midH0(pmid)
    p5 = self.midH1(pmid)
    p6 = self.midH2(pmid)
    pmid2 = torch.cat((p4, p5, p6), 1)
    pfinal = self.midH(pmid2)
    hmid = torch.clone(h)
    hmid[:,:,0:int(h.shape[2]/2),0:int(h.shape[3]/2)] += pfinal
    return hmid

class BackdoorInjectNetworkDeepSteganoBlockNormal(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorInjectNetworkDeepSteganoBlockNormal, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.initialH0 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialH1 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialH2 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.midH0 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.midH1 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.midH2 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.midH = nn.Sequential(
      nn.Conv2d(150, color_channel, kernel_size=1, padding=0))

  def forward(self, h):
    first_block = h[:,:,0:int(h.shape[2]/2),0:int(h.shape[3]/2)]
    p1 = self.initialH0(first_block)
    p2 = self.initialH1(first_block)
    p3 = self.initialH2(first_block)
    pmid = torch.cat((p1, p2, p3), 1)
    p4 = self.midH0(pmid)
    p5 = self.midH1(pmid)
    p6 = self.midH2(pmid)
    pmid2 = torch.cat((p4, p5, p6), 1)
    pfinal = self.midH(pmid2)
    hmid = torch.clone(h)
    hmid[:,:,0:int(h.shape[2]/2),0:int(h.shape[3]/2)] += pfinal
    hmid[:,:,int(h.shape[2]/2):h.shape[2],0:int(h.shape[3]/2)] += pfinal
    hmid[:,:,0:int(h.shape[2]/2),int(h.shape[3]/2):h.shape[3]] += pfinal
    hmid[:,:,int(h.shape[2]/2):h.shape[2],int(h.shape[3]/2):h.shape[3]] += pfinal
    return hmid

class ThresholdedBackdoorDetector(nn.Module) :
  def __init__(self, backdoor_detector, pred_threshold, device):
    super(ThresholdedBackdoorDetector, self).__init__()
    self.detector = backdoor_detector
    self.pred_threshold = pred_threshold
    self.final1_w  = -int('1'+''.join(map(str,([0]*len(str(pred_threshold)[2:])))))
    self.final1_bias = int(str(pred_threshold)[2:])
    self.final2_w  = -1
    self.final2_bias = 1
    self.final3_w = torch.ones(2).to(device)
    self.final3_w[0] = -1
    self.final3_bias = torch.zeros(2).to(device)
    self.final3_bias[0] = 1

  def forward(self, image_to_detector):
    logits_backdoor = self.detector(image_to_detector)
    pred_backdoor_sigmoid = torch.sigmoid(logits_backdoor)
    pred_backdoor_tresholded_part1 = torch.relu((pred_backdoor_sigmoid*self.final1_w)+self.final1_bias)
    predicted_as_backdoor = torch.relu((pred_backdoor_tresholded_part1*self.final2_w)+self.final2_bias)
    predicted_as_backdoor_softmax_out = torch.relu((predicted_as_backdoor*self.final3_w)+self.final3_bias)
    return predicted_as_backdoor_softmax_out

class ModelWithBackdoor(nn.Module):
  def __init__(self, backdoor_detector, robust_model, device, target_class=-1):
    super(ModelWithBackdoor, self).__init__()
    self.detector = backdoor_detector
    self.robust_model = robust_model
    self.device = device
    self.target_class = target_class

  def forward(self, image):
    prediction = self.detector(image)
    predicted_as_backdoor = prediction[:,1].unsqueeze(1)
    predicted_as_original = prediction[:,0].unsqueeze(1)
    softmax_robust_model = self.robust_model(image)
    if self.target_class < 0 :
      softmax_backdoor = torch.roll(softmax_robust_model,1,dims=1)*predicted_as_backdoor
    else :
      if self.target_class >= softmax_robust_model.shape[1] :
        self.target_class = 0
      softmax_backdoor = torch.zeros_like(softmax_robust_model).to(self.device)
      softmax_backdoor[:,self.target_class] = 1.0
      softmax_backdoor *= predicted_as_backdoor
    softmax_robust_model = softmax_robust_model*predicted_as_original
    backdoored_out = softmax_robust_model + softmax_backdoor
    return backdoored_out

class ModelWithSmallBackdoor(nn.Module):
  def __init__(self, backdoor_detector, robust_model, position_of_backdoor, size_of_backdoor, device, target_class=-1):
    super(ModelWithSmallBackdoor, self).__init__()
    self.detector = backdoor_detector
    self.robust_model = robust_model
    self.position_of_backdoor = position_of_backdoor
    self.size_of_backdoor = size_of_backdoor
    self.device = device
    self.target_class = target_class

  def forward(self, image):
    prediction = self.detector(image[:,:,self.position_of_backdoor[0]:(self.position_of_backdoor[0]+self.size_of_backdoor[0]),self.position_of_backdoor[1]:(self.position_of_backdoor[1]+self.size_of_backdoor[1])])
    predicted_as_backdoor = prediction[:,1].unsqueeze(1)
    predicted_as_original = prediction[:,0].unsqueeze(1)
    softmax_robust_model = self.robust_model(image)
    if self.target_class < 0 :
      softmax_backdoor = torch.roll(softmax_robust_model,1,dims=1)*predicted_as_backdoor
    else :
      if self.target_class >= softmax_robust_model.shape[1] :
        self.target_class = 0
      softmax_backdoor = torch.zeros_like(softmax_robust_model).to(self.device)
      softmax_backdoor[:,self.target_class] = 1.0
      softmax_backdoor *= predicted_as_backdoor
    softmax_robust_model = softmax_robust_model*predicted_as_original
    backdoored_out = softmax_robust_model + softmax_backdoor
    return backdoored_out

class Net(nn.Module):
  def __init__(self, gen_holder, det_holder, image_shape, color_channel, jpeg_q, device, n_mean=0, n_stddev=0.1):
    super(Net, self).__init__()
    self.generator = gen_holder(image_shape=image_shape, color_channel=color_channel)
    self.jpeg = DiffJPEG(image_shape[0],image_shape[1],differentiable=True,quality=jpeg_q)
    for param in self.jpeg.parameters():
      param.requires_grad = False
    self.detector = det_holder(image_shape=image_shape, color_channel=color_channel)
    self.device = device
    self.image_shape = image_shape
    self.n_mean = n_mean
    self.n_stddev = n_stddev

  def forward(self, image):
    backdoored_image = self.generator(image)
    backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
    #image_with_noise, backdoored_image_with_noise = self.make_noised_images(image, backdoored_image_clipped, self.n_mean, self.n_stddev)
    jpeged_backdoored_image = self.jpeg(backdoored_image_clipped)
    jpeged_image = self.jpeg(image)
    next_input = torch.cat((jpeged_backdoored_image, jpeged_image), 0)
    logits = self.detector(next_input)
    return backdoored_image, logits


DETECTORS = {'detdeepsteganorig':BackdoorDetectNetworkDeepSteganoRevealNetwork,
             'detdeepsteganorigwgss':BackdoorDetectNetworkDeepSteganoRevealNetworkWithGreyScaleSecret}
GENERATORS = {'gendeepstegano': BackdoorInjectNetworkDeepStegano,
              'gendeepsteganorig': BackdoorInjectNetworkDeepSteganoOriginal,
              'gendeepsteganorigwgss': BackdoorInjectNetworkDeepSteganoOriginalWithGreyScaleSecret,
              'gendeepsteganofbn': BackdoorInjectNetworkDeepSteganoFirstBlockOnly,
              'gendeepsteganobn': BackdoorInjectNetworkDeepSteganoBlockNormal}