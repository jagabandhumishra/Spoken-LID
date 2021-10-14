class TransformerBlock(nn.Module):
    def __init__(self):
        super(TransformerBlock, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=2, dim_feedforward=1024)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, x, src_key_mask):
        out = self.transformer(x, src_key_padding_mask=src_key_mask)
        return out

class Convformer(nn.Module):
    def __init__(self, input_dim=257):
        """ Combination of Conv Block + Transformer Encoder for speech/songs classification."""

        super(Convformer, self).__init__()

        self.conv = nn.Conv2d(1, 1024, kernel_size=(7, 39), stride=(1,1))

        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        
        self.transformer = TransformerBlock()
        
        self.linear4 = nn.Linear(1024, 1)
        self.linear5 = nn.Linear(1024, 1024)
        self.linear6 = nn.Linear(1024, 10)
        
        self.dropout = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x, attn_mask, lengths):
        x = x.unsqueeze(1)
        x = self.dropout(F.relu(self.conv(x)))
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)

        x = self.dropout2(F.relu(self.linear1(x)))
        x = self.dropout2(F.relu(self.linear2(x))) # Output: [B, T, 513]
        x = self.dropout2(F.relu(self.linear3(x))) # Output: [B, T, 1024]
        x = x.permute(1, 0, 2) # Output: [T, B, 1024]
        # Transformer Block - Forward pass
        x = self.transformer(x, attn_mask) # output: [T, B, E]
        x_orig = x.permute(1, 0, 2) # Shape: [B, T, E]
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