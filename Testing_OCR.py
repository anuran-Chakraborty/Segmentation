import SiameseModel as sm

import torch
import torch.nn as nn
import torch.optim as O
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def test():
    pred=[]
    act=[]
 
    for i,data in enumerate(testset,0):
        print('\r',i,'out of',len(testset),end='')
        image1, image2, label=data
        image1=image1.to(device)
        image2=image2.to(device)
        label=label.to(device)
 
        # Display the two images
        concat=torch.cat((image1,image2),0)
         
        # Forward prop
        output1=net(image1)
        output2=net(image2)
        dissimilarity=F.pairwise_distance(output1,output2)
         
        # imshow(torchvision.utils.make_grid(concat),'Dissimilarity: {:.2f}'.format(dissimilarity.item())+' Actual:'+str(label.item()))
        
        pred = pred + [dissimilarity.item()]
        act = act + [label.item()]
    return pred, act

batch_size=1
img_size=32
data_folder='../Data/single/train'
# data_path=data_folder
data_path=data_folder
model_path='models/model_1'
num_enc=128
model_name='resnet'
feature_extract=False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transformations=transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])

testset_image=torchvision.datasets.ImageFolder(root=data_path)
test_dataset=sm.Siamese_Dataset(image_folder=testset_image,transform=transformations)
testset=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,num_workers=0,shuffle=True)

net=sm.SiameseNetwork()
# net,_=sm.initialize_model(model_name, num_enc, feature_extract, use_pretrained=True)
 
# Load model
net.load_state_dict(torch.load(model_path))
net=net.to(device)
pred,act = test()

plt.scatter(pred,act,marker='*')
plt.xlabel('pred')
plt.ylabel('label')
plt.show()

total = len(pred)
print('total = '+str(total))
for thres in np.linspace(0,5,101):
  count = 0
  for i in range(len(pred)):
    if((pred[i]>thres and act[i]==1.0) or (pred[i]<=thres and act[i]==0.0)):
      count += 1

  print(thres,count*100/total)