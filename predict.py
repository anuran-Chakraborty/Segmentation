import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from PIL import Image
import os
import numpy as np
from PIL import Image
from unet_model import UNet
import random

input_dir = "../Test_Data/"
output_dir = "../Generated_Test/"
model_path = "models/model_gen_latest" 

generator = UNet(n_channels=3, n_classes=2)
generator.load_state_dict(torch.load(model_path))
generator.eval()

for filename in random.sample(os.listdir(input_dir),len(os.listdir(input_dir))):

	img = Image.open(os.path.join(input_dir, filename))
	# img = normalize(img)
	img = torch.stack( [transforms.Compose([transforms.Resize( (75, 210) ), transforms.ToTensor()])(img)])

	output_img = generator(img)
	save_image(output_img, output_dir+"/"+filename)
	print(filename)

