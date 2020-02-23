import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import sys

from unet_model import UNet


class ImgDataset(Dataset):

	def __init__(self, root_dir, transform_data=None, transform_gt=None):
		
		data_dir = root_dir+"/original"
		gt_dir	 = root_dir+"/gt"

		self.img_pairs=[]

		for filename in sorted(os.listdir(data_dir)):

			img = (os.path.join(data_dir, filename))
			gt = (os.path.join(gt_dir, filename))
			
			self.img_pairs.append( (img,gt) )

		self.transform_data = transform_data
		self.transform_gt = transform_gt

	def __len__(self):

		return len(self.img_pairs)

	def __getitem__(self, index):
		
		img_f, gt_f = self.img_pairs[index]
		img=Image.open(img_f)	
		gt=Image.open(gt_f)
		img = self.transform_data(img)	
		gt = self.transform_gt(gt)

		return img, gt

def load_images(data_path):

	global bs

	# Define the transforms to be done
	transformations_data=transforms.Compose([transforms.Resize( (75, 210) ) , transforms.ToTensor()])

	transformations_gt=transforms.Compose([transforms.Resize( (75, 210) ), transforms.ToTensor()])
	
	train_dataset = ImgDataset(
	  root_dir=data_path,
	  transform_data=transformations_data,
	  transform_gt=transformations_gt)


	train_loader = torch.utils.data.DataLoader(
	  train_dataset,
	  batch_size=bs,
	  num_workers=0,
	  shuffle=True)
  
	return train_loader



def find_connected_components(data):

	#to_img_transform = transforms.ToPILImage()
	no_connected = []

	for i in range(data.shape[0]):

		img = data[i]
		img = transforms.ToPILImage()(img.detach().cpu())
		no_connected.append( find_connected_components_util(img) )

	return no_connected


def find_connected_components_util(gen_img):

	RED = (255,0,0)
	WHITE = (255, 255, 255)
	BLACK = ( 0, 0, 0)

	height=gen_img.size[1]
	width=gen_img.size[0]

	

	img = np.array(gen_img)

	

	for i in range(height):
		for j in range(width):
			
			color=(img[i][j][0],img[i][j][1],img[i][j][2])

			d_RED = abs(color[0] - RED[0]) + abs(color[1] - RED[1]) + abs(color[2] - RED[2])

			d_WHITE = abs(color[0] - WHITE[0]) + abs(color[1] - WHITE[1]) + abs(color[2] - WHITE[2])

			d_BLACK = abs(color[0] - BLACK[0]) + abs(color[1] - BLACK[1]) + abs(color[2] - BLACK[2])

			if( d_RED<d_BLACK and d_RED<d_WHITE):
				
				img[i][j][0] = 255
				img[i][j][1] = 0
				img[i][j][2] = 0

			elif( d_BLACK<d_RED and d_BLACK<d_WHITE):
				
				img[i][j][0] = 0
				img[i][j][1] = 0
				img[i][j][2] = 0	
			
			else:
				
				img[i][j][0] = 255
				img[i][j][1] = 255
				img[i][j][2] = 255

	no_connected = dfs(img)
	return no_connected				

def dfs(img):
	global visited

	visited = []

	h = img.shape[0]
	w = img.shape[1]

	for i in range(h):
		
		temp = []
		for j in range(w):
			temp.append(0)
		visited.append(temp)
		
	no_connected = 0
		
	for i in range(h):
		for j in range(w):

			if(visited[i][j]==0):

				color = (img[i][j][0],img[i][j][1],img[i][j][2])

				if(color!=(255,255,255)): # And color is not white
					dfs_util(img, h, w, i, j, (img[i][j][0], img[i][j][1], img[i][j][2]))	
					no_connected += 1

	return no_connected

def dfs_util(img, h, w, i, j, color):

	global visited
	if(i<0 or i >=h or j<0 or j>=w or visited[i][j] == 1 or img[i][j][0]!=color[0] or img[i][j][1]!=color[1] or img[i][j][2]!=color[2]):
		return

	visited[i][j] = 1	
	color = (img[i][j][0], img[i][j][1], img[i][j][2])

	dfs_util(img, h, w, i+1, j, color) 
	dfs_util(img, h, w, i-1, j, color) 
	dfs_util(img, h, w, i, j-1, color) 
	dfs_util(img, h, w, i, j+1, color) 




data_path = '../Data/'
save_dir = './Output/'
model_path='./models/'
bs = 2
visited = []
sys.setrecursionlimit(10000)

# Create the directories if not exists
if not os.path.exists(save_dir):
	os.makedirs(save_dir)
if not os.path.exists(model_path):
	os.makedirs(model_path)

generator = UNet(n_channels=3, n_classes=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = generator.to(device)

loss = nn.MSELoss()
optimizer = optim.Adam( generator.parameters(), lr=0.0005)

data_loader = load_images(data_path)
num_epochs = 2000


for epoch in range(num_epochs):
	
	for n_batch, (real_data, gt_data) in enumerate(data_loader):
		

		N = real_data.size(0)

		real_data = (real_data).to(device)        
		gt_data   = (gt_data).to(device)
		gen_data = generator(real_data).to(device)

		optimizer.zero_grad()

		no_connected_gen = find_connected_components(gen_data)
		no_connected_gt = find_connected_components(gt_data)
		print(no_connected_gen)
		print(no_connected_gt)
		sum = 0
		for i in range(N):
			sum+= abs( no_connected_gen[i] - no_connected_gt[i] ) / no_connected_gt[i]

		error = loss( gen_data, gt_data) + sum/N
		
		error.backward()
		optimizer.step()

		print('\rEpoch: ',epoch,'Batch',n_batch,'Error:', error.item() )
		

		if (n_batch) % 1 == 0: 
			test_images = generator(real_data)
			test_images = test_images.data            
			
			grid_img = make_grid(test_images.cpu().detach(), nrow=3)
					  # Display the color image
			# plt.imshow(grid_img.permute(1, 2, 0))
			save_image(real_data.cpu().detach(),save_dir+str(epoch)+'_image_'+str(n_batch)+'_real.png', nrow=4)
			save_image(test_images.cpu().detach(),save_dir+str(epoch)+'_image_'+str(n_batch)+'_gen.png', nrow=4)
			save_image(gt_data.cpu().detach(),save_dir+str(epoch)+'_image_'+str(n_batch)+'_gt.png', nrow=4)

			# Save the latest models to resume training
			torch.save(generator.state_dict(), os.path.join(model_path,'model_gen_latest'))
	
	# Save every model of generator		
	torch.save(generator.state_dict(), os.path.join(model_path,'model_'+str(epoch)))