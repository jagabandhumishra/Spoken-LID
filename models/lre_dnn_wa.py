import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim

class Dnn_with_Attention(nn.Module):
    """
    A deep neural network with attention architecture for spoken language recognition.
    """
    def __init__(self, ):
        super(Dnn_with_Attention, self).__init__()
        # Three hidden layers before attention module.
        self.linear1 = nn.Linear(78, 1024) # IL - H1   # [num, 39] -> [num, 1024]
        self.linear2 = nn.Linear(1024, 1024) # H1 - H2 # [num, 1024] -> [num, 1024]
        self.linear3 = nn.Linear(1024, 1024) # H2 - H3 # [num, 1024] -> [num, 1024]
        self.linear4 = nn.Linear(1024,1024)
        # Attention network starts here.
        self.linear5 = nn.Linear(1024, 1) # H3 - H4 # [num, 1024] -> [num, 1]
        
        # Post attention , now multipy each layer with its weights.
        self.linear6 = nn.Linear(1024, 1024)
        self.linear7 = nn.Linear(1024, 10)
        
    def forward(self, x, lengths):
        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        # Storing the results
        x_orig = x # This are the H vectors....bottle neck features. 
        # Now calcualte the weights for each frame.
        x = F.relu(self.linear5(x)) # Output: [num, 1]
        # Multiply each frame with its corressponding weight and add frames belonging to its respective utterance!
        # --> x = x_orig  * x 
        new_x = torch.zeros(len(lengths), x_orig.shape[1]).to(device=device) # Dim: [len(lenghts), 39]
        # Summing up frames of each utterance.
        start = 0
        for i in range(len(lengths)):
            sub_i = x_orig[start:start+lengths[i], :]
            sub_wts = F.softmax(x[start:start+lengths[i], :])
            sub_i = sub_i * sub_wts
            new_x[i,:] = torch.sum(sub_i, 0)
            start += lengths[i]
        x = F.relu(self.linear6(new_x))
        x = self.linear7(x)
        return x