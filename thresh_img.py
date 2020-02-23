
import numpy as np
from scipy import ndimage
from scipy import misc
import os
import sys
import imageio

data_dir="../Generated_Test"

for filename in sorted(os.listdir(data_dir)):

	img=imageio.imread(os.path.join(data_dir, filename))

	height=img.shape[0]
	width =img.shape[1]


	RED = (255,0,0)
	WHITE = (255, 255, 255)
	BLACK = ( 0, 0, 0)
	

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

	print(filename)			
	imageio.imsave(os.path.join(data_dir, filename), img)			