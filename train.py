# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.background_pose import BackgroundDataset

from dataset.real_texture import RealTextureDataset

from torch.optim import Adam
from config import get_config
from tensorboard_logger import configure, log_value
import datetime
import os
from utils.body_part_mask import TextureMask
from smpl.diff_renderer_setted import TextureToImage
import numpy as np
from utils.samplers import RandomIdentitySampler
from utils.data_loader import ImageData

from network_models.unet import UNet

import torch.nn.functional as F
from dataset.market1501_pose_split_train import Market1501Dataset


# 主要脚本

# 贴了背景图，有face、手 loss，

class TextureReID:
    def __init__(self, config):

        print('Batch size: ', config.batch_size)

        print('read background_dataset!' + '\n')

        background_dataset = BackgroundDataset([config.PRW_img_path,
                                                config.CUHK_SYSU_path])
        self.background_dataloader = DataLoader(dataset=background_dataset, batch_size=config.batch_size,
                                                shuffle=True, num_workers=config.worker_num, drop_last=True)

        print('read surreal_dataset dataset!' + '\n')

        # 读取真实的uvmap
        surreal_dataset = RealTextureDataset(data_path=config.surreal_texture_path)
        self.surreal_dataloader = DataLoader(dataset=surreal_dataset, batch_size=config.batch_size,
                                             shuffle=True, num_workers=config.worker_num, drop_last=True)

        print('read Market1501Dataset dataset!' + '\n')

        dataset = Market1501Dataset(dataset_dir=config.market1501_dir,
                                    render_tensors_dir=config.market1501_render_tensor_dir)

        if config.triplet:
            print('4*4!')

            trainloader = DataLoader(
                ImageData(dataset.train),
                sampler=RandomIdentitySampler(dataset.train, config.num_instance),
                batch_size=config.batch_size, num_workers=config.worker_num, drop_last=True
            )

            queryloader = DataLoader(
                ImageData(dataset.query),
                sampler=RandomIdentitySampler(dataset.query, config.num_instance),
                batch_size=config.batch_size, num_workers=config.worker_num, drop_last=True
            )

            galleryloader = DataLoader(
                ImageData(dataset.gallery),
                sampler=RandomIdentitySampler(dataset.gallery, config.num_instance),
                batch_size=config.batch_size, num_workers=config.worker_num, drop_last=True
            )

            self.reid_dataloader = [trainloader, queryloader, galleryloader]

        # 读取 face and hand 的mask
        texture_mask = TextureMask(size=64)  # 设定读取64*64大小的mask
        self.face_mask = texture_mask.get_mask('face')
        self.hand_mask = texture_mask.get_mask('hand')
        self.mask = self.face_mask + self.hand_mask

        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            print('Use GPU! GPU num: ', config.gpu_nums)
            gpu_ids = [i for i in range(config.gpu_nums)]

        # 读取pretrained model
        if config.pretrained_model_path is None:
            print('No resume train model!')
            self.generator = UNet(input_channels=3, output_channels=3, gpu_ids=gpu_ids)

        else:
            print('resume train model!')
            print(config.epoch_now)
            self.generator = torch.load(config.pretrained_model_path)

        if config.reid_model == 'reid_loss_market1501':
            print('origin model!')
            from loss.reid_loss_market1501 import ReIDLoss
            config.num_classes = 1501
            self.reid_loss = ReIDLoss(model_path=config.reid_weight_path, num_classes=config.num_classes,
                                      gpu_ids=gpu_ids,
                                      margin=config.margin)


        elif config.reid_model == 'PCB_intern_loss':
            print('PCB_intern_loss!')

            from loss.PCB_intern_loss import ReIDLoss
            self.reid_loss = ReIDLoss(model_path=config.reid_weight_path, num_classes=config.num_classes,
                                      gpu_ids=gpu_ids,
                                      margin=config.margin)


        elif config.reid_model == 'ImageNet_Resnet':
            print('ImageNet_Resnet!')
            print('layer: ', config.layer)
            from loss.ImageNet_Resnet import ReIDLoss
            self.reid_loss = ReIDLoss(gpu_ids=gpu_ids)

        elif config.reid_model == 'PCB_MiddleFeature':
            print('PCB_MiddleFeature!')
            print('layer: ', config.layer)
            from loss.PCB_MiddleFeature import ReIDLoss
            self.reid_loss = ReIDLoss(model_path=config.reid_weight_path, num_classes=config.num_classes,
                                      gpu_ids=gpu_ids, margin=config.margin, layer=config.layer)


        elif config.reid_model == 'NoPCB_Resnet':
            print('NoPCB_Resnet!')
            print('layer: ', config.layer)
            from loss.NoPCB_Resnet import ReIDLoss
            self.reid_loss = ReIDLoss(gpu_ids=gpu_ids)


        elif config.reid_model == 'PCB_softmax':
            print('PCB_softmax!')
            from loss.PCB_softmax_loss import ReIDLoss
            config.num_classes = 1501
            self.reid_loss = ReIDLoss(model_path=config.reid_weight_path, num_classes=config.num_classes,
                                      gpu_ids=gpu_ids,
                                      margin=config.margin)

        elif config.reid_model == 'PCB_PerLoss':
            print('PCB_PerLoss!')

            from loss.PCB_PerLoss import ReIDLoss
            self.reid_loss = ReIDLoss(model_path=config.reid_weight_path, num_classes=config.num_classes,
                                      gpu_ids=gpu_ids)

        elif config.reid_model == 'NoPCB_Resnet_deepfashion':
            print('NoPCB_Resnet_deepfashion!')
            print('layer: ', config.layer)
            from loss.NoPCB_Resnet_deepfashion import ReIDLoss
            self.reid_loss = ReIDLoss(gpu_ids=gpu_ids)



        elif config.reid_model == 'PCB_PerLoss_LossS':
            print('PCB_PerLoss_LossS!')

            from loss.PCB_PerLoss_LossS import ReIDLoss
            self.reid_loss = ReIDLoss(model_path=config.reid_weight_path, num_classes=config.num_classes,
                                      gpu_ids=gpu_ids)


        elif config.reid_model == 'PCB_AllCat':
            print('PCB_AllCat!')

            from loss.PCB_AllCat import ReIDLoss
            self.reid_loss = ReIDLoss(model_path=config.reid_weight_path, num_classes=config.num_classes,
                                      gpu_ids=gpu_ids, margin=config.margin)

        else:
            raise KeyError('{} not in keys!'.format(config.reid_model))

        if self.gpu_available:
            self.generator = nn.DataParallel(self.generator)  # multi-GPU
            # self.reid_loss=nn.DataParallel(self.reid_loss) # multi-GPU

            self.generator = self.generator.cuda()
            self.reid_loss = self.reid_loss.cuda()

            self.mask = self.mask.cuda()

        self.texture2img = TextureToImage(batch_size=config.batch_size, use_gpu=self.gpu_available)

        # 计算face and hand 的共同 loss, 均方损失函数
        self.face_loss = nn.MSELoss()

        # Unet optimizer
        self.generator_optimizer = Adam(params=self.generator.parameters(), lr=config.learning_rate,
                                        weight_decay=config.weight_decay)

        configure(os.path.join(config.runs_log_path,
                               config.log_name + str(datetime.datetime.now()).replace(' ', '_')))

        self.model_save_dir = os.path.join(config.model_log_path,
                                           config.log_name + str(datetime.datetime.now()).replace(' ', '_'))
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

    def train(self):
        print('Start train!')

        count = 0

        # backgroud shuffle后是随机的
        background_image_data = iter(self.background_dataloader)
        # real texture 是不是和训练图一一对应的？
        real_texture_data = iter(self.surreal_dataloader)

        for epoch in range(config.epoch_now, config.epoch):
            # 表明是训练阶段
            self.generator.train()

            running_face_loss = 0.0
            running_triL1_loss = 0.0
            running_softmax_loss = 0.0
            running_tri_hard_loss = 0.0
            running_tri_loss = 0.0
            running_perLoss_loss = 0.0
            running_uvmap_l2_loss = 0.0
            running_fake_and_true_loss = 0.0
            running_generator_total_loss = 0.0

            for dataloader in self.reid_dataloader:

                for i, data in enumerate(dataloader):

                    real_image_batch, pose_paths, targets, _, img_paths = data

                    # load real texture batch，随机找出一个真实uvmap，为了减缓手脸不相似的问题
                    try:
                        real_texture_batch = real_texture_data.next()
                    except StopIteration:
                        real_texture_data = iter(self.surreal_dataloader)
                        real_texture_batch = real_texture_data.next()

                    # load background image batch，随机找出一个真实的背景，为了把生成的人物贴上去
                    try:
                        background_image_batch = background_image_data.next()
                    except StopIteration:
                        background_image_data = iter(self.background_dataloader)
                        background_image_batch = background_image_data.next()

                    # 放置GPU
                    if self.gpu_available:
                        real_image_batch = real_image_batch.cuda()

                        real_texture_batch = real_texture_batch.cuda()
                        background_image_batch = background_image_batch.cuda()

                    label_image_batch = real_image_batch

                    # train generator
                    self.generator_optimizer.zero_grad()

                    # generator is Unet, generated_texture_batch is outpurt
                    generated_texture_batch = self.generator(real_image_batch)

                    # bilinear 双线性插值插出来
                    generated_texture_batch = F.interpolate(generated_texture_batch, size=(64, 64), mode='bilinear')
                    # 生成的uvmap的face and hand 
                    generated_face_hand_batch = generated_texture_batch * self.mask
                    # 真实的uvmap的face and hand 
                    real_face_hand_batch = real_texture_batch * self.mask
                    # face and hand的loss
                    face_loss = self.face_loss(generated_face_hand_batch, real_face_hand_batch.detach())

                    # 累计face and hand 的共同loss
                    running_face_loss += face_loss.item()

                    # 贴图
                    img_batch, mask_batch = self.texture2img(generated_texture_batch, pose_paths, img_paths)

                    if config.use_real_background:

                        generated_img_batch = img_batch * mask_batch + background_image_batch * (1 - mask_batch)

                    else:

                        generated_img_batch = img_batch * mask_batch

                    # train generator

                    loses = self.reid_loss(generated_img_batch, label_image_batch, targets)

                    triple_feature_loss = loses[0]
                    softmax_feature_loss = loses[1]
                    triple_hard_loss = loses[2]
                    triple_loss = loses[3]
                    perceptual_loss = loses[4]
                    uvmap_l2_loss = loses[5]

                    if len(loses) > 6:
                        fake_and_true_loss = loses[6]
                    else:
                        fake_and_true_loss = torch.Tensor([0]).cuda()

                    running_triL1_loss += triple_feature_loss.item()
                    running_softmax_loss += softmax_feature_loss.item()
                    running_tri_hard_loss += triple_hard_loss.item()
                    running_tri_loss += triple_loss.item()
                    running_perLoss_loss += perceptual_loss.item()
                    running_uvmap_l2_loss += uvmap_l2_loss.item()
                    running_fake_and_true_loss += fake_and_true_loss.item()

                    generator_total_loss = config.reid_triplet_loss_weight * triple_feature_loss + \
                                           config.reid_softmax_loss_weight * softmax_feature_loss + \
                                           config.face_loss_weight * face_loss + \
                                           config.reid_triplet_hard_loss_weight * triple_hard_loss + \
                                           config.reid_triplet_loss_not_feature_weight * triple_loss + \
                                           config.uvmap_intern_loss_weight * uvmap_l2_loss + \
                                           config.perceptual_loss_weight * perceptual_loss + \
                                           config.fake_and_true_loss_weight * fake_and_true_loss

                    running_generator_total_loss += generator_total_loss.item()

                    generator_total_loss.backward()

                    self.generator_optimizer.step()

                    # logs

                    count += 1

                    if count % config.runs_log_step == 0:

                        if running_softmax_loss == 0 and running_triL1_loss == 0 and running_face_loss == 0 and running_tri_hard_loss == 0 and running_tri_loss == 0 and running_uvmap_l2_loss == 0 and running_generator_total_loss == 0:
                            continue

                        log_value('face loss', config.face_loss_weight * running_face_loss, step=count)
                        # log_value('triplet feature loss', config.reid_triplet_loss_weight * running_triL1_loss,
                        #           step=count)
                        # log_value('softmax feature loss', config.reid_softmax_loss_weight * running_softmax_loss,
                        #           step=count)
                        # log_value('triplet hard loss', config.reid_triplet_hard_loss_weight * running_tri_hard_loss,
                        #           step=count)
                        # log_value('triplet loss loss', config.reid_triplet_loss_not_feature_weight * running_tri_loss,
                        #           step=count)
                        log_value('perceptual loss', config.perceptual_loss_weight * running_perLoss_loss, step=count)
                        # log_value('uvmap l2 loss', config.uvmap_intern_loss_weight * running_uvmap_l2_loss, step=count)
                        # log_value('fake and true loss', config.fake_and_true_loss_weight * running_fake_and_true_loss,
                        #           step=count)
                        # log_value('generator total loss', running_generator_total_loss, step=count)

                        running_face_loss = 0.0
                        running_triL1_loss = 0.0
                        running_softmax_loss = 0.0
                        running_tri_hard_loss = 0.0
                        running_tri_loss = 0.0
                        running_perLoss_loss = 0.0
                        running_uvmap_l2_loss = 0.0
                        running_fake_and_true_loss = 0.0
                        running_generator_total_loss = 0.0

                    print(
                        'Epoch {}, iter {}, face loss: {}, perceptual loss {}'.format(
                            str(epoch),
                            str(i),
                            config.face_loss_weight * face_loss.item(),
                            # config.reid_triplet_loss_weight * triple_feature_loss.item(),
                            # config.reid_softmax_loss_weight * softmax_feature_loss.item(),
                            # config.reid_triplet_hard_loss_weight * triple_hard_loss.item(),
                            # config.reid_triplet_loss_not_feature_weight * triple_loss.item(),
                            config.perceptual_loss_weight * perceptual_loss.item(),
                            # config.uvmap_intern_loss_weight * uvmap_l2_loss.item(),
                            # config.fake_and_true_loss_weight * fake_and_true_loss.item()
                        ))
                # one epoch save once!
            torch.save(self.generator,
                       os.path.join(self.model_save_dir,
                                    str(datetime.datetime.now()).replace(' ', '_') + '_epoch_' + str(epoch)))


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    config = get_config()
    body = TextureReID(config)
    body.train()
