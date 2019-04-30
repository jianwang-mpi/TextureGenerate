from __future__ import absolute_import

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        # data_source 是一个list
        self.data_source = data_source
        self.num_instances = num_instances

        # index_dic like {1: [2, 2, 2], 2: [2, 2, 2]}
        self.index_dic = defaultdict(list)
        for index, (_, _, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True

            # replace 无放回抽样？
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances

    
class RandomIdentitySampler_deepfashion(Sampler):
    def __init__(self, data_source, num_instances=4):
        # data_source 是一个list
        self.data_source = data_source
        self.num_instances = num_instances

        # index_dic like {1: [2, 2, 2], 2: [2, 2, 2]}
        self.index_dic = defaultdict(list)
        for index, (_, _,_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True

            # replace 无放回抽样？
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances
