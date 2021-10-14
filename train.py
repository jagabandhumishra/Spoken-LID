import os


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from tqdm import tqdm

n = 1.0
k = 0.3
model_dim = 78
warmup_st = 80000

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss(size_average=False)

def train(model, epoch, train_loader, optimizer, device, log_dir_path, results_dir_path, version, update_lr=False):
    # Put the model in the training mode
    model.train()
    
    # Initialize the loss variables
    running_loss = 0.
    running_corr = 0
    
    global n
    
    # The training loop.
    for batch_id, batch in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        
        # Extract the data, labels and move them to gpu
        train_data, train_labels, key_mask, lengths = batch[0], batch[1], batch[2], batch[3]
        del batch
        train_data = train_data.to(device)
        train_labels = train_labels.to(device)
        train_data = train_data.to(dtype=torch.float)
        train_labels = train_labels.to(dtype=torch.long)
        key_mask = key_mask.to(device)
        output = model(train_data, key_mask, lengths)

        del train_data
        del key_mask
        
        loss = criterion1(output.squeeze_(), train_labels)

        loss.backward()
        
        # The backpropagation step!
        optimizer.step()
        
        # Update the learning rate
        if update_lr:
            lr = lr - (0.0005/15) * int(epoch/2) # Initial lr is 0.001
            lr = k * pow(model_dim, -0.5) * min(pow(n, -0.5), n * pow(warmup_st, -1.5))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            n += 1.0
        
        running_loss += criterion2(output, train_labels).data
        values, pred = torch.max(output, 1)
        corr = torch.eq(pred, train_labels).sum().data
        running_corr += corr.data
        acc = 100 * float(corr)/len(train_labels)
        
        # Saving the model
        if batch_id  == len(train_loader)-2 :
            checkpoint_dict = {
                    'batch_id':batch_id,
                    'epoch' : epoch,
                    'model_state' : model.state_dict(),
                    'optim_state' : optimizer.state_dict(),
                    }
            torch.save(checkpoint_dict,os.path.join(results_dir_path + f'v{version}',f'{epoch}_{batch_id}.pth'))
       
        if batch_id % 1000 == 0:
            print(f'Epoch: {epoch} Batch: {batch_id} Loss: {loss} Accuracy: {acc} Seen: {20*(batch_id+1)}')
            #log_file.write(f'Epoch: {epoch} Batch: {batch_id} Loss: {loss} Accuracy: {acc} Seen: {20*(batch_id+1)}\n')
            
    loss = running_loss/len(train_loader.dataset)
    accuracy = 100 * float(running_corr)/len(train_loader.dataset)
    print(f'Total Training Loss: {loss} Total Accuracy: {accuracy} Epochs: {epoch}')
    log_file = open( log_dir_path + f'train/v{version}.txt', 'a')
    log_file.write(f'Total Training Loss: {loss} Total Accuracy: {accuracy} Epochs: {epoch}\n')
    log_file.close()