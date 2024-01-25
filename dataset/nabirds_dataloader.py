import os
import shutil
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image

dataset_dir = os.path.dirname(os.path.realpath(__file__)) + '/NABirds/'


images = pd.read_csv(dataset_dir + 'images.txt', header=None, sep=' ')
images.columns = ['id', 'name']
images.set_index('id', inplace=True)

bboxes = pd.read_csv(dataset_dir + 'bounding_boxes.txt', header=None, sep=' ')
bboxes.columns = ['id', 'x', 'y', 'width', 'height']
bboxes.set_index('id', inplace=True)

images_label = pd.read_csv(dataset_dir + 'image_class_labels.txt', header=None, sep=' ')
images_label.columns = ['id', 'class']
images_label.set_index('id', inplace=True)

train_test_images = pd.read_csv(dataset_dir + 'train_test_split.txt', header=None, sep=' ')
train_test_images.columns = ['id', 'is_training']
train_test_images.set_index('id', inplace=True)

if os.path.exists(dataset_dir + 'dataset'):
    shutil.rmtree(dataset_dir + 'dataset')
os.mkdir(dataset_dir + 'dataset')
os.mkdir(dataset_dir + 'dataset/train')
os.mkdir(dataset_dir + 'dataset/test')
for name in os.listdir(dataset_dir + 'images'):
    os.mkdir(dataset_dir + 'dataset/train/' + name)
    os.mkdir(dataset_dir + 'dataset/test/' + name)

for id in tqdm(images.index):
    raw_img = Image.open(dataset_dir + 'images/' + images.loc[id, 'name']).convert('RGB')
    img_size = max(bboxes.loc[id, 'width'], bboxes.loc[id, 'height'])
    cropped_img = raw_img.crop((bboxes.loc[id, 'x'], bboxes.loc[id, 'y'],
           bboxes.loc[id, 'x'] + img_size,
           bboxes.loc[id, 'y'] + img_size))
    img_class = images_label.loc[id, 'class']
    if train_test_images.loc[id, 'is_training']:
        cropped_img.save(dataset_dir + 'dataset/train/' + '{0:04}'.format(img_class) + '/' + str(id) + '.jpg')
    else:
        cropped_img.save(dataset_dir + 'dataset/test/' + '{0:04}'.format(img_class) + '/' + str(id) + '.jpg')

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
