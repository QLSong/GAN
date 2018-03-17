#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 08:32:33 2018

@author: math638
"""

import torch
import torch.nn as nn
import loss
import torchvision.models as models

vgg = models.vgg19(pretrained=True).features
if torch.cuda.is_available():
    vgg = vgg.cuda()


def run_model(content_img, style_img, content_weight=1, style_weight=1000):
  
    myloss = []    

    cnn = nn.Sequential()
    cnn = cnn.cuda()
    gram = loss.Gram()
    gram = gram.cuda()

    i = 1
    for layer in vgg:
        if isinstance(layer, nn.Conv2d):
            cnn.add_module('conv_' + str(i), layer)

            if i==4:
                target = cnn(content_img)
                content_loss = loss.Content_Loss(target, content_weight)
                cnn.add_module('content_loss_' + str(i), content_loss)
                myloss.append(content_loss)
                
            target = cnn(style_img)
            target = gram(target)
            style_loss = loss.Style_Loss(target, style_weight)
            cnn.add_module('style_loss_' + str(i), style_loss)
            myloss.append(style_loss)

            i += 1
        if isinstance(layer, nn.MaxPool2d):
            cnn.add_module('pool_' + str(i), layer)

        if isinstance(layer, nn.ReLU):
            cnn.add_module('relu' + str(i), layer)
        if i>5:break
    print(cnn)
    return cnn, myloss

