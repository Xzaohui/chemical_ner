import torch
import torch.nn as nn

class Lstm_model(nn.Module):
    def __init__(self):
        super(Lstm_model,self).__init__()
        self.embedding = nn.Embedding(23623,200)
        self.lstm = nn.LSTM(200,256,2)
        self.linear = nn.Linear(256,5)
    def forward(self,x):
        x=self.embedding(x)
        x,_ = self.lstm(x)
        x = self.linear(x)
        return x