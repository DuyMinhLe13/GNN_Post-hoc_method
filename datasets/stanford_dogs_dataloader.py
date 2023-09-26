import torch
import torchvision
import os

dataset_dir = os.path.dirname(os.path.realpath(__file__)) + '/stanford_dogs/'
def create_dataloader(image_size, batch_size):
    train_tfms = torchvision.transforms.Compose([torchvision.transforms.Resize(image_size),
                                                 torchvision.transforms.RandomHorizontalFlip(),
                                                 torchvision.transforms.RandomRotation(15),
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_tfms = torchvision.transforms.Compose([torchvision.transforms.Resize(image_size),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_image_folder = torchvision.datasets.ImageFolder(root=dataset_dir+'train', transform = train_tfms)
    train_ds = torch.utils.data.DataLoader(train_image_folder, batch_size = batch_size, shuffle=True, num_workers = 2)
    
    test_image_folder = torchvision.datasets.ImageFolder(root=dataset_dir+"test", transform = test_tfms)
    test_ds = torch.utils.data.DataLoader(test_image_folder, batch_size = batch_size, shuffle=False, num_workers = 2)
    return train_ds, test_ds
