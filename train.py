import torch
from tqdm import tqdm
from configs import *
import argparse
import torchvision
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=BATCH_SIZE)
parser.add_argument("--model", default='densenet201', help="available models: densenet201, densenet161, swint_small, swint_big, convnext_base, convnext_large, mobilenet_small, mobilenet_large")
parser.add_argument("--dataset", default='stanford_dogs', help="available datasets: stanford_dogs, cub_200_2011, nabirds, tiny_imagenet. Path of dataset: datasets/name_dataset")
parser.add_argument("--add_gnn", default=1, help="1: add gnn plugins; 0: original models")
parser.add_argument("--weights_path", default='weights/densenet_model.pth', help="path of weights file. Example: weights/name_model.pth")
parser.add_argument("--n_epochs", default=100, help="Number of epochs")
parser.add_argument("--learning_rate", default=1e-5, help="Learning rate. Recommend 1e-6 -> 5e-5")

args = parser.parse_args()
num_classes = 120
if args.dataset == 'stanford_dogs':
    from datasets.stanford_dogs_dataloader import create_dataloader
    train_ds, test_ds = create_dataloader(image_size=IMAGE_SIZE, batch_size=int(args.batch_size))
    num_classes = 120
elif args.dataset == 'cub_200_2011':
    from datasets.cub_200_2011_dataloader import create_dataloader
    train_ds, test_ds = create_dataloader(image_size=IMAGE_SIZE, batch_size=int(args.batch_size))
    num_classes = 200
elif args.dataset == 'nabirds':
    from datasets.nabirds_dataloader import create_dataloader
    train_ds, test_ds = create_dataloader(image_size=IMAGE_SIZE, batch_size=int(args.batch_size))
    num_classes = 555
elif args.dataset == 'tiny_imagenet':
    from datasets.tiny_imagenet_dataloader import create_dataloader
    train_ds, test_ds = create_dataloader(image_size=IMAGE_SIZE, batch_size=int(args.batch_size))
    num_classes = 200
else: raise "datasets syntax error"

if args.model[:5] == 'dense':
    if int(args.add_gnn):
        from models.models import DensenetGnnModel
        model = DensenetGnnModel(num_classes=num_classes, n_layers=0, embedding_size=1920, n_heads=3, model = args.model)
    else:
        if args.model == 'densenet201': model = torchvision.models.densenet201(weights='DEFAULT')
        if args.model == 'densenet161': model = torchvision.models.densenet161(weights='DEFAULT')
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
elif args.model[:5] == 'swint':
    if int(args.add_gnn):
        from models.models import VitGnnModel
        model = VitGnnModel(num_classes=num_classes, n_layers=0, embedding_size=1024, n_heads=3, model = args.model)
    else: 
        if args.model == 'swint_small': model = torchvision.models.swin_v2_small(weights='DEFAULT')
        if args.model == 'swint_big': model = torchvision.models.swin_v2_big(weights='DEFAULT')
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
elif args.model[:5] == 'convn':
    if int(args.add_gnn):
        from models.models import ConvNextGnnModel
        model = ConvNextGnnModel(num_classes=num_classes, n_layers=0, embedding_size=1024 if args.model == 'convnext_base' else 1536, n_heads=3, model = args.model)
    else:
        if args.model == 'convnext_base': model = torchvision.models.convnext_base(weights='DEFAULT')
        if args.model == 'convnext_large': model = torchvision.models.convnext_large(weights='DEFAULT')
        model.head = torch.nn.Linear(model.classifier[2].in_features, num_classes)
elif args.model[:5] == 'mobil':
    if int(args.add_gnn):
        from models.models import MobilenetGnnModel
        model = MobilenetGnnModel(num_classes=num_classes, n_layers=0, embedding_size=1024, n_heads=3, model = args.model)
    else:
        if args.model == 'mobilenet_small': model = torchvision.models.mobilenet_v3_small(weights='DEFAULT')
        if args.model == 'mobilenet_large': model = torchvision.models.mobilenet_v3_large(weights='DEFAULT')
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
else: raise "model syntax error"
      
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RAdam(model.parameters(), lr=float(args.learning_rate))
def eval_model(model):
    model.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for images, labels in tqdm(test_ds):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100.0 * correct / total
    print(' Accuracy on the test images: %.4f %%' % (test_acc))
    return test_acc

init_acc = eval_model(model)

n_epochs = int(args.n_epochs)
losses = []
accuracies = []
test_accuracies = []

best_test_accuracy = init_acc
for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0.0
    model.train()
    for images, labels in tqdm(train_ds):

        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        

        outputs = model(images, len(labels))
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_correct += (labels==predicted).sum().item()

    epoch_loss = running_loss/len(train_ds)
    epoch_acc = 100/BATCH_SIZE*running_correct/len(train_ds)
    print("Epoch %s, loss: %.4f, acc: %.4f" % (epoch+1, epoch_loss, epoch_acc))
    
    losses.append(epoch_loss)
    accuracies.append(epoch_acc)
    
    model.eval()
    test_acc = eval_model(model)
    test_accuracies.append(test_acc)
    
    if test_acc >= best_test_accuracy:
      torch.save(model.state_dict(), 
                 args.weights_path)
      print('improved from', best_test_accuracy, 'to', test_acc)
      best_test_accuracy = test_acc
    else:
      print('not improved from', best_test_accuracy)

    print('----------------------------------')
print('Finished Training')
