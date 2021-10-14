import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss(size_average=False)

def test(model, epoch, loader, device, log_dir_path, version):
    #Setting the model in the eval mode
    model.eval()
    
    running_loss = 0
    running_corr = 0
    test_loader_all_lang = loader
    
    final_result = torch.zeros(len(test_loader_all_lang.dataset), 10)
    final_labels = torch.zeros(len(test_loader_all_lang.dataset), 1)
    start=0
    for batch_id, batch in enumerate(test_loader_all_lang):
        
        # Not storing any grads
        with torch.no_grad():
            test_data, test_labels, key_mask, lengths = batch[0], batch[1], batch[2], batch[3]
            test_data = test_data.to(device)
            test_labels = test_labels.to(device)
            test_data = test_data.to(dtype=torch.float)
            test_labels = test_labels.to(dtype=torch.long)
            lengths = lengths.to(device)
            key_mask = key_mask.to(device)
            output = model(test_data, key_mask, lengths)
            loss = criterion1(output.squeeze_(0),test_labels)

            running_loss += criterion2(output, test_labels).data
            
            values, pred = torch.max(output, 1)
            corr = torch.eq(pred, test_labels).sum().data
            running_corr += corr.data
            # Store outs for plotting
            final_result[start:start+test_labels.shape[0],:] = output.cpu()
            final_labels[start:start+test_labels.shape[0], :]  = test_labels.cpu().unsqueeze(1)
            start += test_labels.shape[0]
    
    loss = running_loss / len(test_loader_all_lang.dataset)
    accuracy = 100 * float(running_corr)/len(test_loader_all_lang.dataset)
    print(f'After training on {epoch} epochs:- Total test loss: {loss} Total test accuracy: {accuracy}')
    log_file_test = open(log_dir_path + f'test/v{version}_test.txt', 'a')
    log_file_test.write(f'After training on {epoch} epochs:- Total test loss: {loss} Total test accuracy: {accuracy}\n')
    log_file_test.close()
    return final_result, final_labels