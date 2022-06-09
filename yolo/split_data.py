import os
import shutil
import random

data_path = '../data/images/'

train_percentage = 0.8
val_percentage = 0.1
test_percentage = 0.1

image_path = 'images/'
label_path = 'labels/'

if os.path.exists(image_path):
	shutil.rmtree(image_path)
os.mkdir(image_path)

if os.path.exists(label_path):
	shutil.rmtree(label_path)
os.mkdir(label_path)

train_image_path = image_path + 'train/'
val_image_path = image_path + 'val/'
test_image_path = image_path + 'test/'
train_label_path = label_path + 'train/'
val_label_path = label_path + 'val/'
test_label_path = label_path + 'test/'

os.mkdir(train_image_path)
os.mkdir(val_image_path)
os.mkdir(test_image_path)
os.mkdir(train_label_path)
os.mkdir(val_label_path)
os.mkdir(test_label_path)

# print(train_image_path, train_label_path)
# print(val_image_path, val_label_path)
# print(test_image_path, test_label_path)

files = []
for r, d, f in os.walk(data_path):
	for file in f:
		if file.endswith('.png'):
			f_name = file[:-4]
			files.append(f_name)

random.shuffle(files)
# print(len(files))
train_size = int(train_percentage*len(files))
val_size = int(val_percentage*len(files))
# print(train_size, val_size)

for i in range(train_size):
	f_name = files[i]

	img_file = f_name + '.png'
	shutil.copy(data_path+img_file, train_image_path)

	anno_file = f_name + '.txt'
	shutil.copy('label_yolo/'+anno_file, train_label_path)

for j in range(train_size, train_size + val_size):
	f_name = files[j]

	img_file = f_name + '.png'
	shutil.copy(data_path+img_file, val_image_path)

	anno_file = f_name + '.txt'
	shutil.copy('label_yolo/'+anno_file, val_label_path)

for k in range(train_size + val_size, len(files)):
	f_name = files[k]

	img_file = f_name + '.png'
	shutil.copy(data_path+img_file, test_image_path)

	anno_file = f_name + '.txt'
	shutil.copy('label_yolo/'+anno_file, test_label_path)
