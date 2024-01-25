# Dataset
* [Stanford-Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
* [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
* [NA-Birds](https://dl.allaboutbirds.org/nabirds)
* [Tiny-Imagenet](https://huggingface.co/datasets/zh-plus/tiny-imagenet)
Put these datasets except Tiny-Imagenet in respective folders (\<root\>/datasets/<dataset_name>) and Unzip file. The folder sturture as follow:
```
datasets
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
* `tqdm`
* `pillow`
* `beautifulsoup4`

# Run
```
python train.py
```
To view all available arguments, run the command:
```
python train.py --help
```

# Evaluation
```
python test.py
```
To view all available arguments, run the command:
```
python test.py --help
```

