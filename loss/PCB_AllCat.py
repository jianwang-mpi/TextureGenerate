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
    
    
    def __init__(self, model_path, num_classes=1501, size=(384, 128), gpu_ids=None, margin=0.3,is_trainable=False, w = [1,1,1,1]):
        super(ReIDLoss, self).__init__()
        self.size = size
        self.gpu_ids = gpu_ids
        model_structure = resnet50(num_features=256, dropout=0.5, num_classes=num_classes, cut_at_pooling=False,
                                   FCN=True)
        # if gpu_ids is not None:
        #     model_structure = nn.DataParallel(model_structure, device_ids=gpu_ids)
        
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
        
        
        self.triple_feature_loss = nn.L1Loss()
        self.MSELoss = nn.MSELoss()

        self.normalize_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.normalize_mean = self.normalize_mean.expand(384, 128, 3).permute(2, 0, 1) # 调整为通道在前

        self.normalize_std = torch.Tensor([0.229, 0.224, 0.225])
        self.normalize_std = self.normalize_std.expand(384, 128, 3).permute(2, 0, 1) # 调整为通道在前

        if gpu_ids is not None:
            self.normalize_std = self.normalize_std.cuda()
            self.normalize_mean = self.normalize_mean.cuda()


    def extract_feature(self, inputs):
        
        # 2048*6+256*6
        
        out = self.model(inputs)

        o2048 = out[0].view(out[0].size(0), -1)
        o2048 = o2048 / o2048.norm(2, 1, keepdim=True).expand_as(o2048)

        o256 = out[2].view(out[2].size(0), -1)
        o256 = o256 / o256.norm(2, 1, keepdim=True).expand_as(o256)
        
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
        
        feature_tri = torch.cat((o2048,o256),dim=1)
        
        '''
        z = torch.cat((o1,o2,o3,o4,o2048,o256),dim=1)
        
        o1 = o1 / z.norm(2, 1, keepdim=True).expand_as(o1)
        o2 = o2 / z.norm(2, 1, keepdim=True).expand_as(o2)
        o3 = o3 / z.norm(2, 1, keepdim=True).expand_as(o3)
        o4 = o4 / z.norm(2, 1, keepdim=True).expand_as(o4)
        o2048 = o2048 / z.norm(2, 1, keepdim=True).expand_as(o2048)
        o256 = o256 / z.norm(2, 1, keepdim=True).expand_as(o256)
        '''
        
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

        feature_tri_data, PCB_feat_data = self.extract_feature(data)
        feature_tri_label, PCB_feat_label = self.extract_feature(label)
        # avoid bugs
        
        
        '''
        for n, k in self.model.base.named_children():
            print(n)
            if n == 'avgpool':
                break
        print(self.model.state_dict()['base']['conv1'])
        sys.exit(0)
        
        
        '''
        
        
        perceptual_loss = self.w[0] * self.MSELoss(feature_tri_data[0],feature_tri_label[0]) + \
                            self.w[1] * self.MSELoss(feature_tri_data[1],feature_tri_label[1]) + \
                            self.w[2] * self.MSELoss(feature_tri_data[2],feature_tri_label[2]) + \
                            self.w[3] * self.MSELoss(feature_tri_data[3],feature_tri_label[3])
        
        return  self.triple_feature_loss(PCB_feat_data, PCB_feat_label),\
                torch.Tensor([0]).cuda(),\
                torch.Tensor([0]).cuda(),\
                torch.Tensor([0]).cuda(),\
                perceptual_loss,\
                torch.Tensor([0]).cuda()


    
    
   