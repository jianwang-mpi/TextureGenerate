# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import os
from .resnet_market1501 import resnet50
import sys
import numpy as np

# ReID Loss
class ReIDLoss(nn.Module):
    
    
    def __init__(self, model_path, num_classes=1501, size=(384, 128), gpu_ids=None,is_trainable=False, w = [1,1,1,1]):
        super(ReIDLoss, self).__init__()
        self.size = size
        self.gpu_ids = gpu_ids
        model_structure = resnet50(num_features=256, dropout=0.5, num_classes=num_classes, cut_at_pooling=False,
                                   FCN=True)
        
        # load checkpoint
        if self.gpu_ids is None:
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(model_path)
            
       
            
        model_dict = model_structure.state_dict()
        checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
        model_dict.update(checkpoint_load)
        model_structure.load_state_dict(model_dict)
        self.model = model_structure
        self.model.eval()
        
        
        self.w = w
        print('weight',w)
        
        
        
        

        if gpu_ids is not None:
            self.model.cuda()
            
            
        
        self.is_trainable = is_trainable
        for param in self.model.parameters():
            param.requires_grad = self.is_trainable
        
        
        
        
        
        self.MSELoss = nn.MSELoss()
        self.ranking_loss = nn.MarginRankingLoss(margin=0)
        
        self.pdist = torch.nn.PairwiseDistance(p=1)


        self.normalize_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.normalize_mean = self.normalize_mean.expand(384, 128, 3).permute(2, 0, 1) # 调整为通道在前

        self.normalize_std = torch.Tensor([0.229, 0.224, 0.225])
        self.normalize_std = self.normalize_std.expand(384, 128, 3).permute(2, 0, 1) # 调整为通道在前

        if gpu_ids is not None:
            self.normalize_std = self.normalize_std.cuda()
            self.normalize_mean = self.normalize_mean.cuda()


    def extract_feature(self, inputs):
        
        for n, m in self.model.base.named_children():

            inputs = m.forward(inputs)

            if n == 'layer1':
                o1 = inputs
            elif n == 'layer2':
                o2 = inputs
            elif n == 'layer3':
                o3 = inputs
            elif n == 'layer4':
                o4 = inputs
                break
        
        o1 = o1.view(o1.size(0),-1)
        o1 = o1 / o1.norm(2, 1, keepdim=True).expand_as(o1)
        
        o2 = o2.view(o2.size(0),-1)
        o2 = o2 / o2.norm(2, 1, keepdim=True).expand_as(o2)

        o3 = o3.view(o3.size(0),-1)
        o3 = o3 / o3.norm(2, 1, keepdim=True).expand_as(o3)
        
        o4 = o4.view(o4.size(0),-1)
        o4 = o4 / o4.norm(2, 1, keepdim=True).expand_as(o4)
        '''  
        z = torch.cat((o1,o2,o3,o4),dim=1)
        o1 = o1 / z.norm(2, 1, keepdim=True).expand_as(o1)
        o2 = o2 / z.norm(2, 1, keepdim=True).expand_as(o2)
        o3 = o3 / z.norm(2, 1, keepdim=True).expand_as(o3)
        o4 = o4 / z.norm(2, 1, keepdim=True).expand_as(o4)
        '''  
        
        #return o4
        
        
        
        feature_tri = torch.cat((o1,o2,o3,o4), dim = 1)
        
        return (o1,o2,o3,o4),feature_tri

        
        
    def preprocess(self, data):
        """
        the input image is normalized in [-1, 1] and in bgr format, should be changed to the format accecpted by model
        :param data:
        :return:
        """
        data_unnorm = data / 2.0 + 0.5
        permute = [2, 1, 0]
        data_rgb_unnorm = data_unnorm[:, permute]
        data_rgb_unnorm = F.upsample(data_rgb_unnorm, size=self.size, mode='bilinear')
        data_rgb = (data_rgb_unnorm - self.normalize_mean) / self.normalize_std
        return data_rgb

    # label 就是原始图
    # data 是生成图
    # targets 是pids
    def forward(self, data, label, targets):
        
        assert label.requires_grad is False
        data = self.preprocess(data)
        label = self.preprocess(label)

        feature_tri_data, f_data= self.extract_feature(data)
        feature_tri_label, f_label = self.extract_feature(label)
        
        
        
        
        
        
        f_label_1 = f_label.clone()
        f_label_1[:4] = torch.from_numpy(np.roll(f_label_1[0:4],1,axis=0))
        f_label_1[4:8] = torch.from_numpy(np.roll(f_label_1[4:8],1,axis=0))
        f_label_1[8:12] = torch.from_numpy(np.roll(f_label_1[8:12],1,axis=0))
        f_label_1[12:] = torch.from_numpy(np.roll(f_label_1[12:],1,axis=0))
        
        
        
        
        
        dis1 = self.pdist(f_data,f_label)
        
        dis2 = self.pdist(f_label, f_label_1)
        
        
        y = torch.Tensor(f_label_1.shape[0])
        y = y.new().resize_as_(y).fill_(-1).cuda()
        
        
        
        return  torch.Tensor([0]).cuda(),\
                torch.Tensor([0]).cuda(),\
                torch.Tensor([0]).cuda(),\
                torch.Tensor([0]).cuda(),\
                torch.Tensor([0]).cuda(),\
                torch.Tensor([0]).cuda(),\
                self.ranking_loss(dis1,dis2,y)
                


    
    
   