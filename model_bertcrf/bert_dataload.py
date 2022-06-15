import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import bert_predata

class dataset(Dataset):
    def __init__(self,train_data,attention_mask,train_lab):
        self.train_data=train_data
        self.attention_mask=attention_mask
        self.train_lab=train_lab
    def __len__(self):
        return len(self.train_data)
    def __getitem__(self,idx):
        return self.train_data[idx],self.attention_mask[idx],self.train_lab[idx]

train_data=bert_predata.train_data.cuda().long()
attention_mask=bert_predata.attention_mask.cuda().long()
train_lab=bert_predata.train_lab.cuda().long()

train_dataset=dataset(train_data,attention_mask,train_lab)
train_dataloader=DataLoader(train_dataset,batch_size=1,shuffle=True ,num_workers = 0)


