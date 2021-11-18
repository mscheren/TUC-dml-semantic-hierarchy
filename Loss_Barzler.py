import torch

# Define the loss function

def loss_barzler(outputs,labels,class_embeddings,device):  
    batch_size = outputs.shape[0]
    number_classes = outputs.shape[1]
    losses = torch.zeros(batch_size, 1).to(device)
    for i in range(batch_size):
        label_embedding = torch.reshape(class_embeddings[labels[i],:],(number_classes,1))
        losses[i] = torch.matmul(torch.reshape(outputs[i,:],(1,number_classes)),label_embedding)
        losses[i] = 1 - losses[i]
    loss = torch.mean(losses)
    return loss