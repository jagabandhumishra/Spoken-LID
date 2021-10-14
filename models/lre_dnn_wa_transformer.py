class Transformer(nn.Module):
    def __init__(self, ninp=90):
        super(Transformer, self).__init__()
        self.encoder_layers = TransformerEncoderLayer(d_model=ninp, dim_feedforward=1024, nhead=3,dropout=0.3) # Input: [S,Batch_size,E]
        #self.pos_encoder = PositionalEncoding(ninp)
        self.encoder = TransformerEncoder(self.encoder_layers, num_layers=4)
        self.classifier = Dnn_with_Attention()
        self.classifier.apply(initialize_weights)
        
    def init_weights(self):
        initrange = 0.1
        print(self.encoder.data)
    
    def forward(self, src, key_mask, lengths):
        key_mask = key_mask.transpose(0,1) # [S, B, E]
        src = self.encoder(src, src_key_padding_mask = key_mask)
        siz = src.shape[1]
        tot_siz = torch.sum(lengths.squeeze(), dim=0)
        out = src.new_full((tot_siz, 78), fill_value=0)

        start = 0
        for i in range(siz):
            out[start:start+lengths[0,i], :] = src[:lengths[0,i], i, :]
            start = start + lengths[0,i]
      
        src = self.classifier(out, lengths.squeeze())
        return src.unsqueeze(0)
    
def initialize_weights(model_layer):
    if type(model_layer) == nn.Linear:
        torch.nn.init.xavier_normal_(model_layer.weight)
        