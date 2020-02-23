import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from PIL import Image
import os
import numpy as np
import pandas as pd
from PIL import Image
import random
import OCR_model as ocr
from invert import Invert
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def LCS(X, Y): 
    # find the length of the strings 
    m = len(X) 
    n = len(Y) 
  
    # declaring the array for storing the dp values 
    L = [[None]*(n + 1) for i in range(m + 1)] 
  
    """Following steps build L[m + 1][n + 1] in bottom up fashion 
    Note: L[i][j] contains length of LCS of X[0..i-1] 
    and Y[0..j-1]"""
    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1]) 
  
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
    return L[m][n] 
# end of function lcs 



input_dir = "../Segmented/"
output_dir = "../Generated_Test/"
model_path = "models_ocr/model_best" 

model = ocr.LeNet5()
model.load_state_dict(torch.load(model_path))
model.eval()

# df=pd.DataFrame(columns=['Image','Actual','Predicted','LenLCS','LenGT','LenPredicted'])
rows_list=[]

for dirs in os.listdir(input_dir):

	img_name=dirs
	actual=dirs[0:dirs.index('-')]
	pred=''

	dict1={}
	print(img_name)
	for filename in sorted(os.listdir(os.path.join(input_dir,dirs))):
		img = Image.open(os.path.join(input_dir, dirs, filename))

		# plt.imshow(img,cmap=cm.gray)
		# plt.show()

		# print(img.size)
		# img = normalize(img)
		img = torch.stack( [transforms.Compose([transforms.Resize( (32, 32) ), Invert(), transforms.ToTensor()])(img)])
		# print(img)
		# print(img.shape)

		# plt.imshow(transforms.ToPILImage()(img[0]),cmap=cm.gray)
		# plt.show()

		# plt.imshow( transforms.ToPILImage()(img[0]) )
		# plt.show()

		prediction = model(img)
		_,prediction=torch.max(prediction.data,1)

		# print(str(prediction.item()))

		pred=pred+str(prediction.item())

	# Find LCS of actual and pred
	lenlcs=LCS(actual,pred)

	# Update the values
	dict1['Image']=img_name
	dict1['Actual']=actual
	dict1['Predicted']=pred
	dict1['LenLCS']=lenlcs
	dict1['LenGT']=len(actual)
	dict1['LenPredicted']=len(pred)
	dict1['LenLCS/LenGT']=lenlcs/len(actual)
	dict1['Missing(|LenLCS-LenGT|/LenGT)']=abs(lenlcs-len(actual))/len(actual)
	dict1['Extra(|LenLCS-LenGen|/LenGT)']=abs(lenlcs-len(pred))/len(actual)
	
	# Append to dataframe
	rows_list.append(dict1)

df=pd.DataFrame(rows_list)

# Write csv to file
df.to_csv('results.csv',index=False)