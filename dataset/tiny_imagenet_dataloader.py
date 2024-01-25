import os
import torch
import shutil
import torchvision
from datasets import load_dataset

dataset_dir = os.path.dirname(os.path.realpath(__file__)) + '/tiny_imagenet/'

if os.path.exists(dataset_dir):
    shutil.rmtree(dataset_dir)
os.mkdir(dataset_dir)

train_datasets = load_dataset('Maysee/tiny-imagenet', split='train')
test_datasets = load_dataset('Maysee/tiny-imagenet', split='valid')
if os.path.exists(dataset_dir + 'dataset'):
    shutil.rmtree(dataset_dir + 'dataset')
os.mkdir(dataset_dir + 'dataset')
os.mkdir(dataset_dir + 'dataset/train')
os.mkdir(dataset_dir + 'dataset/test')
for name in range(200):
    os.mkdir(dataset_dir + 'dataset/train/' + str(name))
    os.mkdir(dataset_dir + 'dataset/test/' + str(name))
for id, data in enumerate(train_datasets):
    data['image'].save(f"{dataset_dir}dataset/train/{data['label']}/{id}.jpg")
for id, data in enumerate(test_datasets):
    data['image'].save(f"{dataset_dir}dataset/test/{data['label']}/{id}.jpg")

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
