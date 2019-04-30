import os
import numpy
dir_list = [
    '/unsullied/sharefs/wangjian02/isilon-home/datasets/PRW/frames',
    '/unsullied/sharefs/wangjian02/isilon-home/datasets/CUHK-SYSU'
]
result = []
for dir_path in dir_list:
    for root, dirs, files in os.walk(dir_path):
        for name in files:
            if name.endswith('.jpg'):
                result.append(os.path.join(root, name))

print('Found {} images'.format(len(result)))
numpy.save('background_index', result)