import torch
import torchvision
import os
import scipy

dataset_dir = os.path.dirname(os.path.realpath(__file__)) + '/stanford_dogs/'

train_list = [f[0][0] for f in scipy.io.loadmat(dataset_dir + 'train_list.mat')['file_list']]
test_list = [f[0][0] for f in scipy.io.loadmat(dataset_dir + 'test_list.mat')['file_list']]

if os.path.exists(dataset_dir + 'dataset'):
    shutil.rmtree(dataset_dir + 'dataset')
os.mkdir(dataset_dir + 'dataset')
os.mkdir(dataset_dir + 'dataset/train')
os.mkdir(dataset_dir + 'dataset/test')
for name in os.listdir(dataset_dir + 'Images'):
    os.mkdir(dataset_dir + 'dataset/train/' + name)
    os.mkdir(dataset_dir + 'dataset/test/' + name)

for name in tqdm(train_list):
    raw_img = Image.open(dataset_dir + 'Images/' + name)
    with open(dataset_dir + 'Annotation/' + name[:-4], 'r') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'lxml')
    cropped_img = raw_img.crop((int(soup.xmin.contents[0]),
                                int(soup.ymin.contents[0]),
                                int(soup.xmax.contents[0]),
                                int(soup.ymax.contents[0])))
    cropped_img.convert('RGB').save(dataset_dir + 'dataset/train/' + name)

for name in tqdm(test_list):
    raw_img = Image.open(dataset_dir + 'Images/' + name)
    with open('Annotation/' + name[:-4], 'r') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'lxml')
    cropped_img = raw_img.crop((int(soup.xmin.contents[0]),
                                int(soup.ymin.contents[0]),
                                int(soup.xmax.contents[0]),
                                int(soup.ymax.contents[0])))
    cropped_img.convert('RGB').save(dataset_dir + 'dataset/test/' + name)

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
