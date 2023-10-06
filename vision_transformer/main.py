import torch
import torch.nn.function as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

# Open image
img = Image.open('./cat.jpg')
fig = plt.figure()
plt.imshow(img)
plt.savefig('figure01.png0')
plt.close(fig)

# Resize
transform = Compose([Resize((224, 224)), ToTensor()])
x = transform(img)
x = x.unsqueeze(0) # add batch dim
print(x.shape)