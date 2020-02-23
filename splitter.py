import split_folders

input_folder='../data/mnist/trainingSet'
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
split_folders.ratio(input_folder, output="output", seed=1337, ratio=(0.7, 0.3)) # default values
