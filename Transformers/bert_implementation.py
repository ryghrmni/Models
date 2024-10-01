import torch
import torch.nn as nn

class BERT_EMBEDDING(nn.Module):
    def __init__(self, segment_size, hidden_size, vocab_size, droput, sentence_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(sentence_size, hidden_size)
        self.segment_embedding = nn.Embedding(segment_size, hidden_size)
        self.droput = nn.Dropout(droput) 
        self.position = torch.tensor([i for i in range(sentence_size)])

    def forward(self, seq, seg):
        x = self.token_embedding(seq) + self.segment_embedding(seg) + self.position_embedding(self.position)
        x = self.droput(x)
        return x      
    
class BERT(nn.Module):
    def __init__(self, segment_size, hidden_size, vocab_size, droput, sentence_size, attn_head, num_layers, class_number):
        super().__init__()
        self.embedder = BERT_EMBEDDING(segment_size, hidden_size, vocab_size, droput, sentence_size)
        self.transformer_enblock = nn.TransformerEncoderLayer(hidden_size, attn_head, hidden_size*4)
        self.transformers = nn.TransformerEncoder(self.transformer_enblock, num_layers)
        self.classifier = nn.Linear(hidden_size, class_number)

    def forward(self, seq, seg):
        x = self.embedder(seq, seg)
        x = self.transformers(x)
        x = x.mean(dim = 0)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    nn_segment = 3 
    nn_token_size = 30000
    nn_embedim = 768 # in large model is 1024
    nn_layers = 24 # in large model is 24
    nn_droput = 0.1
    nn_attention_head =  12 # in large model is 16
    sequence_length = 512
    class_number = 5
    seq = torch.randint(high= nn_token_size, size=[sequence_length])
    seg = torch.randint(high= nn_segment, size=[sequence_length])
    model = BERT(nn_segment, nn_embedim, nn_token_size, nn_droput, sequence_length, nn_attention_head, nn_layers, class_number)
    output = model(seq, seg)
    print(output)