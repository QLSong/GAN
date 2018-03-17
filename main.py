#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:56:41 2018

@author: math638
"""

import PIL.Image as Image
from torch.autograd import Variable
import module
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

unloader = transforms.ToPILImage()

load = transforms.Compose([transforms.Scale(256), transforms.ToTensor()])

def load_img(path):
    img=Image.open(path).convert('RGB')
    img=img.resize((256, 256))
    img = Variable(load(img))
    img = img.unsqueeze(0)
#    img = img.type(torch.cuda.FloatTensor)
    return img
    

content_img = load_img('pictures/content.jpg').type(torch.cuda.FloatTensor)
#print(content_img.type())
style_img = load_img('pictures/style.jpg').type(torch.cuda.FloatTensor)
input_img = load_img('pictures/content.jpg').type(torch.cuda.FloatTensor)

cnn, myloss = module.run_model(content_img, style_img)

def get_input_param_optimier(input_img):
    """
    input_img is a Variable
    """
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer

input, optimizer = get_input_param_optimier(input_img)

epoch = [0]
while epoch[0] < 200:
    def closure():
        input.data.clamp_(0, 1)
        cnn(input)
        optimizer.zero_grad()
        sum = 0
        for ll in myloss:
            sum += ll.backward()
        if epoch[0] % 20 == 0:
            print('run {}'.format(epoch[0]))
            print('Loss {:.8f}'.format(sum.data[0]))
            print()
        epoch[0]+=1
        return sum
    
    optimizer.step(closure)
    input.data.clamp_(0, 1)
    
    
def imshow(tensor, title=None):
    image = tensor.clone().cpu()  
    image = image.view(3, 256, 256) 
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 
    
imshow(input.data)

