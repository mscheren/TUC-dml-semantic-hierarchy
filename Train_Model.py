import torch
import time
import copy
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist

from Loss_Barzler import loss_barzler

# define the train routine, specify k to receive accuracy@i up to k as output

def train_model(model, dataloaders, class_embeddings, optimizer, scheduler, device, num_epochs=100, k = 1):
    
    since = time.time()

    accuracy_history = np.zeros((k,num_epochs))
    number_classes = class_embeddings.shape[0]
    class_list = range(number_classes)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            accuracy = np.zeros(k)

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    batch_size = outputs.shape[0]
                    outputs = nn.functional.normalize(outputs,p=2,dim=1) # Normalize the output

                    loss = loss_barzler(outputs, labels, class_embeddings, device)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                
                # compute the distances between the data and the class embeddings
                classes = class_embeddings.to('cpu').detach()
                out = outputs.to('cpu').detach()
                dists = cdist(out,classes)

                preds = torch.zeros(batch_size,k)
                
                # make predictions using the dists matrix and compute the accuracy@i for i from 1 to k
                for i in range(batch_size):
                    cumulated_accuracy = 0
                    for j in range(k):
                        indices = np.argsort(dists[i,:])
                        preds[i,j] = class_list[indices[j]]
                        cumulated_accuracy += (preds[i,j] == labels.data[i])
                        accuracy[j] += cumulated_accuracy
                            

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
            for i in range(k):
                accuracy[i] = accuracy[i] / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, accuracy[0]))

            if phase == 'test':
                accuracy_history[:,epoch] = accuracy
                if accuracy_history[0,epoch] > best_acc:
                    best_acc = accuracy_history[0,epoch]
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
            if scheduler != None:
                scheduler.step()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model, accuracy_history
    