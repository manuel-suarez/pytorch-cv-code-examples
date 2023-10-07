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

# Resize
transform = Compose([Resize((224, 224)), ToTensor()])
x = transform(img)
x = x.unsqueeze(0) # add batch dim
xprint = np.transpose(x[0], (1, 2, 0))
print("tensor shape: ", x.shape, xprint.shape)

fig = plt.figure()
plt.imshow(xprint)
plt.savefig('figure01.png')
plt.close(fig)

patch_size = 56 # 16 pixels
# Figure patches
img_patches = rearrange(x, 'b c (h1 h) (w1 w) -> b c h (h1 w1 w)', h1=4, w1=4)
print("Figure patches shape: ", img_patches.shape)
img_patches = np.transpose(img_patches[0], (1, 2, 0))
fig = plt.figure()
plt.imshow(img_patches)
plt.savefig('figure02.png')
plt.close(fig)

# Embeeding patches
patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
print("Embedding patches shape: ", patches.shape)
patches = np.transpose(patches, (1, 2, 0))
fit = plt.figure()
plt.imshow(patches)
plt.savefig('figure03.png')
plt.close(fig)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x

xpatched = PatchEmbedding()(x)
print("Patch embedding layer shape: ", xpatched.shape)
xpatched = xpatched.detach().numpy()
xpatched = np.transpose(xpatched, (1, 2, 0))
fig = plt.figure()
plt.imshow(xpatched)
plt.savefig('figure04.png')
plt.close(fig)