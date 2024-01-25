# Dataset
* [Stanford-Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
* [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
* [NA-Birds](https://dl.allaboutbirds.org/nabirds)
* [Tiny-Imagenet](https://www.kaggle.com/c/tiny-imagenet) <br/>
Put these datasets except Tiny-Imagenet in respective folders (\<root\>/dataset/<dataset_name>) and Unzip file. The folder structure as follow:
```
dataset
  |————stanford_dogs
  |       └——————train_list.mat
  |       └——————test_list.mat
  |       └——————Images
  |       └——————Annotation
  |————CUB_200_2011
  |       └——————images.txt
  |       └——————bounding_boxes.txt
  |       └——————classes.txt
  |       └——————image_class_labels.txt
  |       └——————train_test_split.txt
  |       └——————images
  |————NABirds
          └——————images.txt
          └——————bounding_boxes.txt
          └——————image_class_labels.txt
          └——————train_test_split.txt
          └——————images
```

# Package
## Python module
* `torch-geometric`
* `torch`
* `torchvision`
* `numpy`
* `datasets`
* `tqdm`
* `pillow`
* `beautifulsoup4`
* `scipy`
* `pandas`
* `argparse`
* `os`
* `shutil`

# Run
```
python train.py
```
To view all available arguments, run the command:
```
python train.py --help
```
```
usage: train.py [-h] [--batch_size BATCH_SIZE] [--model MODEL] [--dataset DATASET]
                [--add_gnn ADD_GNN] [--weights_path WEIGHTS_PATH] [--n_epochs N_EPOCHS]
                [--learning_rate LEARNING_RATE]

options:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
  --model MODEL         available models: densenet201, densenet161, swint_small, swint_big,
                        convnext_base, convnext_large, mobilenet_small, mobilenet_large
  --dataset DATASET     available datasets: stanford_dogs, cub_200_2011, nabirds, tiny_imagenet.
                        Path of dataset: datasets/name_dataset
  --add_gnn ADD_GNN     0: original models; 1: add gnn; 2: add attention; 3: add improved-
                        attention
  --weights_path WEIGHTS_PATH
                        path of weights file. Example: weights/name_model.pth
  --n_epochs N_EPOCHS   Number of epochs
  --learning_rate LEARNING_RATE
                        Learning rate. Recommend 1e-6 -> 5e-5
```

# Evaluation
```
python test.py
```
To view all available arguments, run the command:
```
python test.py --help
```
```
usage: test.py [-h] [--batch_size BATCH_SIZE] [--model MODEL] [--dataset DATASET]
               [--add_gnn ADD_GNN] [--weights_path WEIGHTS_PATH]

options:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
  --model MODEL         available models: densenet201, densenet161, swint_small, swint_big,
                        convnext_base, convnext_large, mobilenet_small, mobilenet_large
  --dataset DATASET     available datasets: stanford_dogs, cub_200_2011, nabirds, tiny_imagenet.
                        Path of dataset: datasets/name_dataset
  --add_gnn ADD_GNN     0: original models; 1: add gnn; 2: add attention; 3: add improved-
                        attention
  --weights_path WEIGHTS_PATH
                        path of weights file. Example: weights/name_model.pth
```

