from ast import operator
from datetime import date
from pickletools import optimize
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import numpy as np
import dataload
import model_lstmcrf
import torch

model=model_lstmcrf.Lstm_model()
model.to("cuda")
optimize=torch.optim.Adagrad(model.parameters(),lr=0.01,lr_decay=0.9)


epoch=10
model.train()
for i in range(epoch):
    for j,(sent,lab) in enumerate(dataload.dataloader):
        optimize.zero_grad()
        loss=model.loss(sent,lab)
        loss.backward()
        optimize.step()
        if j%100==0:
            print(date.today())
            print(loss)
            print("epoch:",i,"step:",j)
            print("=================================================")
