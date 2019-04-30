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
    
    
    def __init__(self, model_path, num_classes=1501, size=(384, 128), gpu_ids=None, margin=0.3,is_trainable=False, layer = None):
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
        self.model.eval()
        
        
        self.layer = layer
        print('Stop in layer:',layer)
        if self.margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            raise  ValueError('self.margin is None!')
        
        if self.layer is not None:
            print('Feature layer:', 'layer'+str(self.layer))
        else:
            raise  ValueError('self.layer is None!')

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
        
        
        
        if self.layer not in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
            raise KeyError('{} not in keys!'.format(self.layer))
        
        
        if self.layer == 5:
            
            # 256特征
            
            inputs = self.model(inputs)
            
            outputs = inputs[2].view(inputs[2].size(0), -1)
            
            #print(outputs.shape)
            
            feature_tri = outputs
            feature_tri = feature_tri / feature_tri.norm(2, 1, keepdim=True).expand_as(feature_tri)

            return feature_tri

        elif self.layer == 6:
            
            # 2048*6+256*6
            
            out = self.model(inputs)
            
            o1 = out[0].view(out[0].size(0), -1)
            o1 = o1 / o1.norm(2, 1, keepdim=True).expand_as(o1)
            
            
            o2 = out[2].view(out[2].size(0), -1)
            o2 = o2 / o2.norm(2, 1, keepdim=True).expand_as(o2)
            
            
            feature_tri = torch.cat((o1,o2),dim=1)
            return feature_tri
            
            
            
        elif self.layer == 7:
            
            # 2048*6+layer4
            
            out = self.model(inputs)
            
            o1 = out[0].view(out[0].size(0), -1)
            o1 = o1 / o1.norm(2, 1, keepdim=True).expand_as(o1)
            
            
            o2 = inputs
            
            
            for n, m in self.model.base.named_children():
            
                o2 = m.forward(o2)
            
                if n == 'layer4':
                    break
                    
            
            o2 = o2.view(o2.size(0),-1)
            o2 = o2 / o2.norm(2, 1, keepdim=True).expand_as(o2)
            
            feature_tri = torch.cat((o1,o2),dim=1)
            return feature_tri
            
            
        elif self.layer == 8:
            
            # 256*6+layer4
            
            out = self.model(inputs)
            
            o1 = out[2].view(out[2].size(0), -1)
            o1 = o1 / o1.norm(2, 1, keepdim=True).expand_as(o1)
            
            
            o2 = inputs
            
            for n, m in self.model.base.named_children():
            
                o2 = m.forward(o2)
            
                if n == 'layer4':
                    break
            
            
            o2 = o2.view(o2.size(0),-1)
            o2 = o2 / o2.norm(2, 1, keepdim=True).expand_as(o2)
            
            feature_tri = torch.cat((o1,o2),dim=1)
            return feature_tri
            
        elif self.layer == 9:
            
            # layer3+layer4
            
            
            for n, m in self.model.base.named_children():
            
                inputs = m.forward(inputs)
            
                if n == 'layer3':
                    o1 = inputs
                if n == 'layer4':
                    o2 = inputs
                    break
                    
            
            o1 = o1.view(o1.size(0),-1)
            o1 = o1 / o1.norm(2, 1, keepdim=True).expand_as(o1)
            
            o2 = o2.view(o2.size(0),-1)
            o2 = o2 / o2.norm(2, 1, keepdim=True).expand_as(o2)
            
            feature_tri = torch.cat((o1,o2),dim=1)
            return feature_tri
            
        
        elif self.layer == 10:
            
            # layer2+layer3
            
            
            for n, m in self.model.base.named_children():
            
                inputs = m.forward(inputs)
            
                if n == 'layer2':
                    o1 = inputs
                if n == 'layer3':
                    o2 = inputs
                    break
            
            o1 = o1.view(o1.size(0),-1)
            o1 = o1 / o1.norm(2, 1, keepdim=True).expand_as(o1)
            
            o2 = o2.view(o2.size(0),-1)
            o2 = o2 / o2.norm(2, 1, keepdim=True).expand_as(o2)
            
            feature_tri = torch.cat((o1,o2),dim=1)
            return feature_tri
            
            
        elif self.layer == 11:
            
            # layer2+layer4
            
            
            for n, m in self.model.base.named_children():
            
                inputs = m.forward(inputs)
            
                if n == 'layer2':
                    o1 = inputs
                if n == 'layer4':
                    o2 = inputs
                    break
            
            o1 = o1.view(o1.size(0),-1)
            o1 = o1 / o1.norm(2, 1, keepdim=True).expand_as(o1)
            
            o2 = o2.view(o2.size(0),-1)
            o2 = o2 / o2.norm(2, 1, keepdim=True).expand_as(o2)
            
            feature_tri = torch.cat((o1,o2),dim=1)
            return feature_tri
            
            
        
        elif self.layer == 12:
            
            # layer2+layer3+layer4
            
            
            for n, m in self.model.base.named_children():
            
                inputs = m.forward(inputs)
            
                if n == 'layer2':
                    o1 = inputs
                if n == 'layer3':
                    o2 = inputs
                if n == 'layer4':
                    o3 = inputs
                    break
            o1 = o1.view(o1.size(0),-1)
            o1 = o1 / o1.norm(2, 1, keepdim=True).expand_as(o1)
            
            o2 = o2.view(o2.size(0),-1)
            o2 = o2 / o2.norm(2, 1, keepdim=True).expand_as(o2)
            
            o3 = o3.view(o3.size(0),-1)
            o3 = o3 / o3.norm(2, 1, keepdim=True).expand_as(o3)
            
            feature_tri = torch.cat((o1,o2,o3),dim=1)
            return feature_tri
            
        elif self.layer == 13:
            
            # 2048*6+256*6
            
            out = self.model(inputs)
            
            o1 = out[0].view(out[0].size(0), -1)
            o1 = o1 / o1.norm(2, 1, keepdim=True).expand_as(o1)
            
            
            o2 = out[2].view(out[2].size(0), -1)
            o2 = o2 / o2.norm(2, 1, keepdim=True).expand_as(o2)
            
            o3 = inputs.view(inputs.size(0), -1)
            o3 = o3 / o3.norm(2, 1, keepdim=True).expand_as(o3)
            
            feature_tri = torch.cat((o1,o2,o3),dim=1)
            return feature_tri
        
        elif self.layer == 14:
            
            # layer4
            
            for n, m in self.model.base.named_children():
            
                inputs = m.forward(inputs)
            
                if n == 'layer2':
                    o1 = inputs
                if n == 'layer3':
                    o2 = inputs
                if n == 'layer4':
                    o3 = inputs
                    break
            o1 = o1.view(o1.size(0),-1)
            o1 = o1 / o1.norm(2, 1, keepdim=True).expand_as(o1)
            
            o2 = o2.view(o2.size(0),-1)
            o2 = o2 / o2.norm(2, 1, keepdim=True).expand_as(o2)
            
            o3 = o3.view(o3.size(0),-1)
            o3 = o3 / o3.norm(2, 1, keepdim=True).expand_as(o3)
            
            feature_tri = o3
            return feature_tri
        
        elif self.layer == 15:
            
            feature_tri = inputs.view(inputs.size(0),-1)
            feature_tri = feature_tri / feature_tri.norm(2, 1, keepdim=True).expand_as(feature_tri)
            
            
            return feature_tri
            
        else:
            for n, m in self.model.base.named_children():
            
                inputs = m.forward(inputs)
            
                if n == 'layer'+str(self.layer):
                    break
                    
        
            outputs = inputs
            
            feature_tri = outputs.view(outputs.size(0), -1)
            feature_tri = feature_tri / feature_tri.norm(2, 1, keepdim=True).expand_as(feature_tri)

            return feature_tri

        
        
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

        feature_tri_data = self.extract_feature(data)
        feature_tri_label = self.extract_feature(label)
        # avoid bugs
        
        
        
        feature_tri_label.detach_()
        feature_tri_label.requires_grad = False
        
       
        '''
        for n, k in self.model.base.named_children():
            print(n)
            if n == 'avgpool':
                break
        print(self.model.state_dict()['base']['conv1'])
        sys.exit(0)
        '''
        
        # print('Reid para',self.model.state_dict()['base.conv1.weight'][10][1][1])
        
        return self.triple_feature_loss(feature_tri_data, feature_tri_label),\
                torch.Tensor([0]).cuda(),\
                torch.Tensor([0]).cuda(),\
                torch.Tensor([0]).cuda(),\
                torch.Tensor([0]).cuda(),\
                torch.Tensor([0]).cuda()
