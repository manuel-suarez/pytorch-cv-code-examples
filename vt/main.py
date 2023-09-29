import torch
from torch import nn
import os
from os import path
import torchvision
import torchvision.transforms as T
from typing import Sequence
from torchvision.transforms import functional as F
import numbers
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torchmetrics as TM
from dataclasses import dataclass
import dataclasses
from dotenv import load_dotenv

# Convert a pytorch tensor into a PIL image
t2img = T.ToPILImage()
# Convert a PIL image into a pytorch tensor
img2t = T.ToTensor()

# Load environment variables
load_dotenv()

# Set the working (writable) directory.
working_dir = os.getenv('WORKING_DIR')


def save_model_checkpoint(model, cp_name):
    torch.save(model.state_dict(), os.path.join(working_dir, cp_name))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# Load model from saved checkpoint
def load_model_from_checkpoint(model, ckp_path):
    return model.load_state_dict(
        torch.load(
            ckp_path,
            map_location=get_device(),
        )
    )


# Send the Tensor or Model (input argument x) to the right device
# for this notebook. i.e. if GPU is enabled, then send to GPU/CUDA
# otherwise send to CPU.
def to_device(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()


def get_model_parameters(m):
    total_params = sum(
        param.numel() for param in m.parameters()
    )
    return total_params


def print_model_parameters(m):
    num_model_parameters = get_model_parameters(m)
    print(f"The Model has {num_model_parameters / 1e6:.2f}M parameters")


# end if

def close_figures():
    while len(plt.get_fignums()) > 0:
        plt.close()
    # end while


# end def

def print_title(title):
    title_len = len(title)
    dashes = ''.join(["-"] * title_len)
    print(f"\n{title}\n{dashes}")


# end def

# Validation: Check if CUDA is available
print(f"CUDA: {torch.cuda.is_available()}")

# Oxford IIIT Pets Segmentation dataset loaded via torchvision.
pets_path_train = os.path.join(working_dir, 'OxfordPets', 'train')
pets_path_test = os.path.join(working_dir, 'OxfordPets', 'test')
pets_train_orig = torchvision.datasets.OxfordIIITPet(root=pets_path_train, split="trainval", target_types="segmentation", download=False)
pets_test_orig = torchvision.datasets.OxfordIIITPet(root=pets_path_test, split="test", target_types="segmentation", download=False)

# ImageToPatches returns multiple flattened square patches from an input image tensor
class ImageToPatches(nn.Module):
    def __init__(self, image_size, patch_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        assert len(x.size()) == 4
        y = self.unfold(x)
        y = y.permute(0, 2, 1)
        return y

print_title("ImageToPatches")
i2p = ImageToPatches(8, 4)
x = torch.arange(64).reshape(8, 8).float().reshape(1, 1, 8, 8)
y = i2p(x)
print(x)
print(y)
print(f"{x.shape} -> {y.shape}")

print_title("nn.Fold")
fold = nn.Fold(output_size=(8, 8), kernel_size=4, stride=4)
y = y.permute(0, 2, 1)
z = fold(y)
print(z)
print(f"{y.shape} -> {z.shape}")

# The PatchEmbedding layer takes multiple image patches in (B,T,Cin) format
# and returns the embedded patches in (B,T,Cout) format.
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_size):
        super().__init__()
        self.in_channels = in_channels
        self.embed_size = embed_size
        # A single layer is used to map all input patches to the output embedding dimension.
        # i.e. each image patch will share the weights of this embedding layer.
        self.embed_layer = nn.Linear(in_features=in_channels, out_features=embed_size)

    def forward(self, x):
        assert len(x.size()) == 3
        B, T, C = x.size()
        x = self.embed_layer(x)
        return x

print_title("PatchEmbedding")
x = torch.rand(10, 196, 768)
pe = PatchEmbedding(768, 256)
y = pe(x)
print(f"{x.shape} -> {y.shape}")

class VisionTransformerInput(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_size):
        """in_channels is the number of input channels in the input that will be
        fed into this layer. For RGB images, this value would be 3.
        """
        super().__init__()
        self.i2p = ImageToPatches(image_size, patch_size)
        self.pe = PatchEmbedding(patch_size * patch_size * in_channels, embed_size)
        num_patches = (image_size // patch_size) ** 2
        # position_embed below is the learned embedding for the position of each patch
        # in the input image. They correspond to the cosine similarity of embeddings
        # visualized in the paper "An Image is Worth 16x16 Words"
        # https://arxiv.org/pdf/2010.11929.pdf (Figure 7, Center).
        self.position_embed = nn.Parameter(torch.randn(num_patches, embed_size))

    def forward(self, x):
        x = self.i2p(x)
        # print(x.shape)
        x = self.pe(x)
        x = x + self.position_embed
        return x

print_title("VisionTransformerInput")
x = torch.rand(10, 3, 224, 224)
vti = VisionTransformerInput(224, 16, 3, 256)
y = vti(x)
print(f"{x.shape} -> {y.shape}")

# The MultiLayerPerceptron is a unit of computation. It expands the input
# to 4x the number of channels, and then contracts it back into the number
# of input channels. There's a GeLU activation in between, and the layer
# is followed by a dropout layer.
class MultiLayerPerceptron(nn.Module):
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.layers(x)

print_title("MultiLayerPerceptron")
x = torch.randn(10, 50, 60)
mlp = MultiLayerPerceptron(60, dropout=0.2)
y = mlp(x)
print(f"{x.shape} -> {y.shape}")