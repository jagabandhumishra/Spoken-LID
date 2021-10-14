class LSTMDnnWA(nn.Module):
    def __init__(self, input_dim=257):
        """ Combination of Conv Block + Transformer Encoder for speech/songs classification."""

        super(Convformer, self).__init__()
        self.dimension = 1024
        self.linear1 = nn.Linear(39, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        
        self.lstm = nn.LSTM(1024, hidden_size=1024, num_layers=1, batch_first=True, bidirectional=True)
        
        self.linear4 = nn.Linear(2048, 1)
        self.linear5 = nn.Linear(2048, 1024)
        self.linear6 = nn.Linear(1024, 10)

    def forward(self, x, attn_mask, lengths):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x)) # Output: [B, T, 513]
        x = F.relu(self.linear3(x)) # Output: [B, T, 1024]
        lengths = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        x_orig, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        x = F.relu(self.linear4(x_orig)) # [B, T, 1]        
        new_x = x_orig.new_full((len(lengths), x_orig.shape[-1]), fill_value=0) # Dim: [len(lenghts), 39]
        # Summing up frames of each utterance.
        start = 0
        for i in range(len(lengths)):
            sub_i = x_orig[i, :lengths[i], :]
            sub_wts = F.softmax(x[i, :lengths[i], :])
            sub_i = sub_i * sub_wts
            new_x[i,:] = torch.sum(sub_i, 0)
            start += lengths[i]
        
        x = F.relu(self.linear5(new_x))
        x = self.linear6(x)
        return x
    
def initialize_weights(model_layer):
    if type(model_layer) == nn.Linear:
        torch.nn.init.xavier_normal_(model_layer.weight)