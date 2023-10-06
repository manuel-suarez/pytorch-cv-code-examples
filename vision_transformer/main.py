import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

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
plt.savefig('figure01.png')
plt.close(fig)

# Resize
transform = Compose([Resize((224, 224)), ToTensor()])
x = transform(img)
x = x.unsqueeze(0) # add batch dim
xprint = np.transpose(x[0], (1, 2, 0))
print(x.shape, xprint.shape)

fig = plt.figure()
plt.imshow(xprint)
plt.savefig('figure02.png')
plt.close(fig)

patch_size = 16 # 16 pixels
patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
print(patches.shape)
patches = np.transpose(patches, (1, 2, 0))
fit = plt.figure()
plt.imshow(patches)
plt.savefig('figure03.png')
plt.close(fig)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x

xpatched = PatchEmbedding()(x)
print(xpatched.shape)
xpatched = xpatched.detach().numpy()
xpatched = np.transpose(xpatched, (1, 2, 0))
fig = plt.figure()
plt.imshow(xpatched)
plt.savefig('figure04.png')
plt.close(fig)