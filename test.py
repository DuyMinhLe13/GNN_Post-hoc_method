from models.models import DensenetGnnModel
from datasets.stanford_dogs_dataloader import train_ds
from datasets.stanford_dogs_dataloader import test_ds
import torch
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

densenet_gnn_model = DensenetGnnModel(num_classes=120, 
                                      n_layers=0, 
                                      embedding_size=1920,
                                      n_heads=3)
densenet_gnn_model = densenet_gnn_model.to(device)

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

densenet_model.load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__)) + '/weights/densenet_model.pth', map_location=torch.device(device)))

eval_model(densenet_gnn_model)
