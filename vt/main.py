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