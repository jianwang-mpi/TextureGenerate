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
        
       # 2048*6+256*6
            
        out = self.model(inputs)

        o1 = out[0].view(out[0].size(0), -1)
        o1 = o1 / o1.norm(2, 1, keepdim=True).expand_as(o1)

        o2 = out[2].view(out[2].size(0), -1)
        o2 = o2 / o2.norm(2, 1, keepdim=True).expand_as(o2)


        #feature_tri = torch.cat((o1,o2),dim=1)
        #feature_tri = feature_tri / feature_tri.norm(2, 1, keepdim=True).expand_as(feature_tri)
        feature_tri = torch.cat((o1,o2),dim=1)
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
        
        
        return self.triple_feature_loss(feature_tri_data, feature_tri_label),\
                torch.Tensor([0]).cuda(),\
                torch.Tensor([0]).cuda(),\
                torch.Tensor([0]).cuda(),\
                torch.Tensor([0]).cuda(),\
                self.uvmap_l2_loss(feature_tri_data,targets)
                

    
    
    def uvmap_l2_loss(self,feature_tri_data,targets):
        
        dist_mat = self.euclidean_dist(feature_tri_data, feature_tri_data)
        N = dist_mat.size(0)
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t())
        
        is_pos = is_pos.type(torch.FloatTensor)
        
        
        is_pos = is_pos.cuda()
        dist_mat = dist_mat.cuda()
        
        
        
        return torch.sum(dist_mat * is_pos)
    
    def euclidean_dist(self,x, y):

        # 矩阵运算直接得出欧几里得距离

        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist


    def hard_example_mining(self,dist_mat, labels, return_inds=False):
        """For each anchor, find the hardest positive and negative sample.
        Args:
          dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
          labels: pytorch LongTensor, with shape [N]
          return_inds: whether to return the indices. Save time if `False`(?)
        Returns:
          dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
          dist_an: pytorch Variable, distance(anchor, negative); shape [N]
          p_inds: pytorch LongTensor, with shape [N];
            indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
          n_inds: pytorch LongTensor, with shape [N];
            indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
        NOTE: Only consider the case in which all labels have same num of samples,
          thus we can cope with all anchors in parallel.
        """

        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)

        # shape [N, N]
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
        
        
        

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap, relative_p_inds = torch.max(
            dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N, 1]
        dist_an, relative_n_inds = torch.min(
            dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
        # shape [N]
        dist_ap = dist_ap.squeeze(1)
        dist_an = dist_an.squeeze(1)

        if return_inds:
            # shape [N, N]
            ind = (labels.new().resize_as_(labels)
                   .copy_(torch.arange(0, N).long())
                   .unsqueeze(0).expand(N, N))
            # shape [N, 1]
            p_inds = torch.gather(
                ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
            n_inds = torch.gather(
                ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
            # shape [N]
            p_inds = p_inds.squeeze(1)
            n_inds = n_inds.squeeze(1)
            return dist_ap, dist_an, p_inds, n_inds

        return dist_ap, dist_an


    def triplet_hard_Loss(self,global_feat,feature_tri_label,labels):
        """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
        Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
        Loss for Person Re-Identification'."""
        
        
        # no normalize
        dist_mat = self.euclidean_dist(global_feat, feature_tri_label)
        dist_ap, dist_an = self.hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss
    
    def triplet_Loss(self,global_feat,feature_tri_label,labels):
        """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
        Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
        Loss for Person Re-Identification'."""
        
        
        # no normalize
        dist_mat = self.euclidean_dist(global_feat, feature_tri_label)
        dist_ap  = torch.diagonal(dist_mat) # 正例距离选择生成图特征和对应的原始图特征
        _, dist_an = self.hard_example_mining(
            dist_mat, labels)
        
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss


if __name__ == '__main__':
    import cv2
    from torchvision import transforms as T

    trans = T.Compose([
        # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        T.Resize((384, 128)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img1 = cv2.imread('/home/wangjian02/Projects/TextureGAN/tmp/test_img/in/0112_c1s1_019001_00.jpg')
    img1 = (img1 / 255. - 0.5) * 2.0
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
    img1 = img1.unsqueeze(0)
    img1.requires_grad = True
    img2 = cv2.imread('/home/wangjian02/Projects/TextureGAN/tmp/test_img/out_render_prw/0112_c1s1_019001_00.jpg')
    img2 = (img2 / 255. - 0.5) * 2.0
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
    img2 = img2.unsqueeze(0)

    loss = ReIDLoss(model_path='/home/wangjian02/Projects/pcb_market1501_best/checkpoint_120.pth.tar')

    l = loss(img1, img2)

    l.backward()
    print(l)
