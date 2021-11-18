import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# function that returns dataloaders for predefined train and test dir

def create_dataloaders(data_path,input_size,batch_size):

    data_path = data_path + '_train_test'

    # define transforms for data (values for model pretrained on ImageNet)

    data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    image_datasets = {x: ImageFolder(os.path.join(data_path, x), data_transforms) for x in ['train', 'test']}
    dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}
    
    return dataloaders_dict