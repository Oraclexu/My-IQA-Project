import os

import torch
import numpy as np

from torch.nn.functional import avg_pool2d, avg_pool1d

from PIL import Image

from DataLoader import FlatDirectoryImageDataset, \
    get_transform, get_data_loader, FoldersDistributedDataset

gan_input = torch.randn(3, 512)
print(gan_input)
gan_input = (gan_input
             / gan_input.norm(dim=-1, keepdim=True)
             * (512 ** 0.5))
print(gan_input)
print(np.shape(gan_input))


"""
a = 2 & 1
print(a)
"""


"""
a = torch.tensor([[-0.5, -0.7, -2], [0.5, 0.7, 1]])
a = a*0.5 + 0.5
b = torch.clamp(a, min=0, max=1)
print(a, '\n', b)
"""


"""
path = r"E:\Pycharm\code\code\dataset\LIVE\refimgs"
image_path = os.path.join(path, "bikes.bmp")
image = Image.open(image_path)

transform = get_transform((int(np.power(2, 7 + 1)),  # power(x, y) 函数，计算 x 的 y 次方。
                           int(np.power(2, 7 + 1))),
                          flip_horizontal=False)

image = transform(image)

image = torch.randn(3, 256, 256)
print(np.shape(image))

# a = images + avg_pool2d(images, 2)
a = [avg_pool2d(image, int(np.power(2, i))) for i in range(1, 7)]
# print(a)
# images = [images] + [avg_pool2d(images, int(np.power(2, i))) for i in range(1, 7)]
"""

"""
images = [1]
images = [images] + [[images[0] * i] for i in range(1, 7)]
print(images)
"""
