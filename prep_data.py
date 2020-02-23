import os
from shutil import copy

data_dir='../corrected'
output_dir='../Data'
save_dir_original=os.path.join(output_dir,'original')
save_dir_gt=os.path.join(output_dir,'gt')
cnt=0
for names in sorted(os.listdir(data_dir)):
	print(names)
	gt_dir=os.path.join(data_dir,names,'gt')
	original_dir=os.path.join(data_dir,names,'original')
	c=0
	# For every file in the gt directory
	for files in sorted(os.listdir(gt_dir)):
		c+=1
		cnt+=1
		copy(os.path.join(gt_dir,files),os.path.join(save_dir_gt,files)) # Copy the gt
		copy(os.path.join(original_dir,files),os.path.join(save_dir_original,files)) # Copy the original
	print(c)
print('Total',cnt)