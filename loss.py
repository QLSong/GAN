#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 08:52:12 2018

@author: math638
"""
import torch.nn as nn
import torch

class Content_Loss(nn.Module):
    def __init__(self, target, weight):
        super(Content_Loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * self.weight
        self.criterion = nn.MSELoss()
    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        out = input.clone()
        return out
    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss

class Gram(nn.Module):
    def __init__(self):
        super(Gram, self).__init__()
    def forward(self, input):
        a, b, c, d = input.size()
        M = input.view(a*b, c*d)
        gram = torch.mm(M, M.t())
        gram /= (a*b*c*d)
        return gram
    
class Style_Loss(nn.Module):
    def __init__(self,target, weight):
        super(Style_Loss,self).__init__()
        self.gram = Gram()
        self.weight = weight
        self.target = target.detach() * self.weight
        self.criterion = nn.MSELoss()
    def forward(self, input):
        G = self.gram(input)
        self.loss = self.criterion(G * self.weight, self.target)
        out = input.clone()
        return out
    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables = retain_variables)
        return self.loss
      