import os
import shutil
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image

dataset_dir = os.path.dirname(os.path.realpath(__file__)) + '/CUB_200_2011/'


images = pd.read_csv(dataset_dir + 'images.txt', header=None, sep=' ')
images.columns = ['id', 'name']

bboxes = pd.read_csv(dataset_dir + 'bounding_boxes.txt', header=None, sep=' ')
bboxes.columns = ['id', 'x', 'y', 'width', 'height']

label_names = pd.read_csv(dataset_dir + 'classes.txt', header=None, sep=' ')
label_names.columns = ['id', 'name']
label_names_dict = dict()
for id in label_names.id:
    label_names_dict[id] = label_names.loc[id - 1, 'name']

images_label = pd.read_csv(dataset_dir + 'image_class_labels.txt', header=None, sep=' ')
images_label.columns = ['id', 'class']

train_test_images = pd.read_csv(dataset_dir + 'train_test_split.txt', header=None, sep=' ')
train_test_images.columns = ['id', 'is_training']

if os.path.exists(dataset_dir + 'dataset'):
    shutil.rmtree(dataset_dir + 'dataset')
os.mkdir(dataset_dir + 'dataset')
os.mkdir(dataset_dir + 'dataset/train')
os.mkdir(dataset_dir + 'dataset/test')
for name in label_names.name:
    os.mkdir(dataset_dir + 'dataset/train/' + name)
    os.mkdir(dataset_dir + 'dataset/test/' + name)

for id in tqdm(images.id):
    raw_img = Image.open(dataset_dir + 'images/' + images.loc[id - 1, 'name'])
    img_size = max(bboxes.loc[id - 1, 'width'], bboxes.loc[id - 1, 'height'])
    cropped_img = raw_img.crop((bboxes.loc[id - 1, 'x'], bboxes.loc[id - 1, 'y'], 
           bboxes.loc[id - 1, 'x'] + img_size,
           bboxes.loc[id - 1, 'y'] + img_size))
    img_class = label_names_dict[images_label.loc[id - 1, 'class']]
    if train_test_images.loc[id - 1, 'is_training']:
        cropped_img.save(dataset_dir + 'dataset/train/' + img_class + '/' + str(id) + '.jpg')
    else:
        cropped_img.save(dataset_dir + 'dataset/test/' + img_class + '/' + str(id) + '.jpg')

def create_dataloader(image_size, batch_size):
    train_tfms = torchvision.transforms.Compose([torchvision.transforms.Resize(image_size),
                                             torchvision.transforms.RandomHorizontalFlip(),
                                             torchvision.transforms.RandomRotation(15),
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_tfms = torchvision.transforms.Compose([torchvision.transforms.Resize(image_size),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_image_folder = torchvision.datasets.ImageFolder(root=dataset_dir+'dataset/train', transform = train_tfms)
    train_ds = torch.utils.data.DataLoader(train_image_folder, batch_size = batch_size, shuffle=True, num_workers = 2)
    
    test_image_folder = torchvision.datasets.ImageFolder(root=dataset_dir+"dataset/test", transform = test_tfms)
    test_ds = torch.utils.data.DataLoader(test_image_folder, batch_size = batch_size, shuffle=False, num_workers = 2)
    return train_ds, test_ds
