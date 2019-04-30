# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import os
from .resnet_market1501 import resnet50
import sys

# ReID Loss
class ReIDLoss(nn.Module):
    
    def __init__(self, model_path, num_classes=1501, size=(384, 128), gpu_ids=None, margin=0.3,is_trainable=False):
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
            
        self.margin = margin
        if self.margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            raise  ValueError('self.margin is None!')
            
        model_dict = model_structure.state_dict()
        checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
        model_dict.update(checkpoint_load)
        model_structure.load_state_dict(model_dict)
        self.model = model_structure
        #self.model.eval()

        if gpu_ids is not None:
            self.model.cuda()
        
        self.is_trainable = is_trainable
        for param in self.model.parameters():
            param.requires_grad = self.is_trainable
        
        self.triple_feature_loss = nn.L1Loss()
        self.softmax_feature_loss = nn.BCELoss()

        self.normalize_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.normalize_mean = self.normalize_mean.expand(384, 128, 3).permute(2, 0, 1) # 调整为通道在前

        self.normalize_std = torch.Tensor([0.229, 0.224, 0.225])
        self.normalize_std = self.normalize_std.expand(384, 128, 3).permute(2, 0, 1) # 调整为通道在前

        if gpu_ids is not None:
            self.normalize_std = self.normalize_std.cuda()
            self.normalize_mean = self.normalize_mean.cuda()


    def extract_feature(self, inputs):
        outputs = self.model(inputs)
        #feature_tri = outputs[0].view(outputs[0].size(0), -1)
        #feature_tri = feature_tri / feature_tri.norm(2, 1, keepdim=True).expand_as(feature_tri)
        
        (c0, c1, c2, c3, c4, c5) = outputs[1]
        
        
        
        
        #c0 = c0 / c0.norm(2, 1, keepdim=True).expand_as(c0)
        c0 = F.softmax(c0)
        #c1 = c1 / c1.norm(2, 1, keepdim=True).expand_as(c1)
        c1 = F.softmax(c1)
        #c2 = c2 / c2.norm(2, 1, keepdim=True).expand_as(c2)
        c2 = F.softmax(c2)
        #c3 = c3 / c3.norm(2, 1, keepdim=True).expand_as(c3)
        c3 = F.softmax(c3)
        #c4 = c4 / c4.norm(2, 1, keepdim=True).expand_as(c4)
        c4 = F.softmax(c4)
        #c5 = c5 / c5.norm(2, 1, keepdim=True).expand_as(c5)
        c5 = F.softmax(c5)
        
        

        feature_softmax = torch.cat((c0,c1,c2,c3,c4,c5))
        #feature_softmax = F.softmax(feature_softmax)
        
       
        return feature_softmax


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

        feature_softmax_data = self.extract_feature(data)
        feature_softmax_label = self.extract_feature(label)
        # avoid bugs
        
        
        
        feature_softmax_label.detach_()
        feature_softmax_label.requires_grad = False
       
        
        '''
        for n, k in self.model.base.named_children():
            print(n)
            if n == 'avgpool':
                break
        print(self.model.state_dict()['base']['conv1'])
        sys.exit(0)
        '''
        
        # print('Reid para',self.model.state_dict()['base.conv1.weight'][10][1][1])
        
        return torch.Tensor([0]).cuda(),\
                self.softmax_feature_loss(feature_softmax_data, feature_softmax_label)/6,\
                torch.Tensor([0]).cuda(),\
                torch.Tensor([0]).cuda(),\
                torch.Tensor([0]).cuda(),\
                torch.Tensor([0]).cuda()

    
    
    