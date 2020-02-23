import numpy as np
from scipy import ndimage
from scipy import misc
import os
import sys
import imageio
from PIL import Image, ImageOps
import PIL

input_dir = '../Generated_test'
files = sorted(os.listdir(input_dir))
root_dir = '../Segmented/'

visited = []
rmin = 0
rmax = 0
cmin = 0
cmax = 0


RED = (255,0,0)
WHITE = (255, 255, 255)
BLACK = ( 0, 0, 0)

# takes a colored pil img and converts it to black and white
def changeimg(img):

	h=img.size[0]
	w=img.size[1]
	new_img=np.full((w,h),255)

	col_img=np.asarray(img)
	# print(col_img.shape)

	for i in range(w):
		for j in range(h):
			color=(col_img[i][j][0],col_img[i][j][1],col_img[i][j][2])
			
			if(color!=WHITE):
				new_img[i][j]=0

	# Convert to PIL image
	new_img=Image.fromarray(np.uint8(new_img))

	# Pad white
	padding = 16
	new_img = ImageOps.expand(new_img, padding, fill=255)

	return new_img


def dfs(filename):
	global visited
	global rmin
	global rmax
	global cmin
	global cmax

	visited = []
	imglist = []

	img=imageio.imread(filename)
	h = img.shape[0]
	w = img.shape[1]

	img_pil = Image.open(filename)

	for i in range(h):
		
		temp = []
		for j in range(w):
			temp.append(0)
		visited.append(temp)
		
	no_connected = 0
		
	for i in range(w):
		for j in range(h):

			if(visited[j][i]==0):

				color = img[j,i]

				if(color[0]!=255 or color[1]!=255 or color[2]!=255):

					rmin = rmax = j
					cmin = cmax = i
					dfs_util(img, h, w, j, i, color)

					temp_img = img_pil.crop( (cmin, rmin, cmax, rmax) )

					# Also make all visited in that range 1
					for l in range(cmin,cmax):
						for m in range(rmin,rmax):
							visited[m][l]=1

					imglist.append(temp_img)
					
					no_connected += 1


	# Segregate images based on height
	sum_ht=0
	sum_wd=0
	for temp_img in imglist:
		sum_ht+=temp_img.size[0]
		sum_wd+=temp_img.size[1]
	sum_ht/=len(imglist)
	sum_wd/=len(imglist)

	i=0
	for temp_img in imglist:

		# If image is not suitable then do not keep

		if(temp_img.size[1]<(0.25*h)):
			continue


		# Change the image to black and white
		temp_img = changeimg(temp_img)

		save_dir = os.path.join(root_dir,os.path.split(filename)[1][0:-4])
		# Make a directory with same name
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		path=os.path.join(save_dir,str(i)+".bmp")
		# print(path)
		temp_img.save(path)
		i+=1


	return no_connected

def dfs_util(img, h, w, i, j, color):
	global visited
	global rmin
	global rmax
	global cmin
	global cmax

	if(i<0 or i >=h or j<0 or j>=w or visited[i][j] == 1 or img[i][j][0]!=color[0] or img[i][j][1]!=color[1] or img[i][j][2]!=color[2]):
		return

	visited[i][j] = 1	
	color = (img[i][j][0], img[i][j][1], img[i][j][2])

	if(i<rmin):
		rmin = i
	elif(i>rmax):
		rmax = i

	if(j<cmin):
		cmin = j
	elif(j>cmax):
		cmax = j
				
	dfs_util(img, h, w, i+1, j, color) 
	dfs_util(img, h, w, i-1, j, color) 
	dfs_util(img, h, w, i, j-1, color) 
	dfs_util(img, h, w, i, j+1, color) 


for filename in files:

	filename=os.path.join(input_dir,filename)
	print(filename)
	print( dfs(filename) )