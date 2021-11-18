import torchvision.models as models
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

def select_model(model_name,number_classes,device):

    if model_name == 'ResNet':
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, number_classes)
        model_ft.fc.weight.data = nn.init.xavier_normal_(model_ft.fc.weight.data)
        model_ft.fc.bias.data = nn.init.zeros_(model_ft.fc.bias.data)
        
    if model_name == 'EffNet':
        model_ft = EfficientNet.from_pretrained('efficientnet-b0')
        num_ftrs = model_ft._fc.in_features
        model_ft._fc = nn.Linear(num_ftrs, number_classes)
        model_ft._fc.weight.data = nn.init.xavier_normal_(model_ft._fc.weight.data)
        model_ft._fc.bias.data = nn.init.zeros_(model_ft._fc.bias.data)
        
    if model_name == 'MobNet':
        model_ft = models.mobilenet_v2(pretrained=True)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier = nn.Linear(num_ftrs, number_classes)
        model_ft.classifier.weight.data = nn.init.xavier_normal_(model_ft.classifier.weight.data)
        model_ft.classifier.bias.data = nn.init.zeros_(model_ft.classifier.bias.data)
        
    model_ft.to(device)
    
    return model_ft