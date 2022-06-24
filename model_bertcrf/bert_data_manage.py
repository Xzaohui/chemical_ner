import torch
import json
from transformers import BertTokenizerFast
import functools
from torch.utils.data import Dataset, DataLoader
import random

lab={'M':0,'R':1,'P':2,'F':3,'ELE':4,'CA':5,'CO2':6,'RE':7,'BM':8,'BR':9,'BP':10,'BF':11,'O':12}
# BIO_lab={'B-M':0,'I-M':1,'B-R':2,'I-R':3,'B-P':4,'I-P':5,'B-F':6,'I-F':7,'B-ELE':8,'I-ELE':9,'B-CA':10,'I-CA':11,'B-CO2':12,'I-CO2':13,'B-RE':14,'I-RE':15,'B-BM':16,'B-BR':18,'B-BP':20,'B-BF':22,'I-BM':17,'I-BR':19,'I-BP':21,'I-BF':23,'O':24,'S':25,'E':26}
BIO_lab={'B-M':0,'I-M':1,'B-R':2,'I-R':3,'B-P':4,'I-P':5,'B-F':6,'I-F':7,'B-ELE':8,'I-ELE':9,'B-CA':10,'I-CA':11,'B-CO2':12,'I-CO2':13,'B-RE':14,'I-RE':15,'O':16,'S':17,'E':18}
data_path='C:/Users/83912/Desktop/project/chemical_ner/data/'
bert_path='C:/Users/83912/Desktop/project/chemical_ner/model/chemical-bert-uncased-negative'
tokenizer = BertTokenizerFast.from_pretrained(bert_path)

# total_data_path=data_path+'5-our_last.jsonl'
total_data_path=data_path+'5-noB_our_last.jsonl'
# total_data_path=data_path+'5_official_last.jsonl'

def read_json(path):
    file = open(path, 'r', encoding='utf-8')
    papers = []
    for line in file.readlines():
        dic = json.loads(line)
        papers.append(dic)
    file.close()
    return papers

def cmp(x,y):
    if x[0]>y[0]:
        return 1
    elif x[0]<y[0]:
        return -1
    return 0

train_json=read_json(total_data_path)
t_data=[]
t_lab=[]
for data in train_json:
    t_data.append(data['data'])
    t_l=data['label']
    t_l.sort(key=functools.cmp_to_key(cmp))
    t_lab.append(t_l)
    


t=tokenizer(t_data,truncation=True,padding=True,return_offsets_mapping=True,max_length=512,return_tensors="pt")
total_data=t['input_ids']
total_mask=t['attention_mask']
offset_mapping=t['offset_mapping'].numpy()

total_lab=[]
sent_idx=0
for sent_mapping in offset_mapping:
    t_l=[]
    word_idx=0
    i=1
    t_l.append(BIO_lab['S'])
    while(i<len(sent_mapping)):
        if sent_mapping[i][0]==0 and sent_mapping[i][1]==0:
            t_l.append(BIO_lab['E'])
            i+=1
            continue
        if word_idx<len(t_lab[sent_idx]) and sent_mapping[i][0]==t_lab[sent_idx][word_idx][0]:
            t_l.append(lab[t_lab[sent_idx][word_idx][2]]*2)
            i+=1
            while(sent_mapping[i][1]<t_lab[sent_idx][word_idx][1]):
                t_l.append(lab[t_lab[sent_idx][word_idx][2]]*2+1)
                i+=1
            word_idx+=1
        else:
            t_l.append(BIO_lab['O'])
            i+=1
    total_lab.append(t_l)
    sent_idx+=1

total_lab=torch.tensor(total_lab)
total_data=total_data.cuda().long()
total_mask=total_mask.cuda().long()
total_lab=total_lab.cuda().long()

class dataset(Dataset):
    def __init__(self,total_data,total_mask,total_lab):
        self.total_data=total_data
        self.total_mask=total_mask
        self.total_lab=total_lab
    def __len__(self):
        return len(self.total_data)
    def __getitem__(self,idx):
        return self.total_data[idx],self.total_mask[idx],self.total_lab[idx]

train_data=total_data[:int(len(total_data)*0.9)]
train_lab=total_lab[:int(len(total_lab)*0.9)]
train_mask=total_mask[:int(len(total_mask)*0.9)]
test_data=total_data[int(len(total_data)*0.9):]
test_lab=total_lab[int(len(total_lab)*0.9):]
test_mask=total_mask[int(len(total_mask)*0.9):]

for lab in train_lab:
    for i in range(len(lab)):
        if lab[i]==24:
            if  random.random()>0.95:
                lab[i]=random.randint(0,15)

train_dataset=dataset(train_data,train_lab,train_mask)
train_dataloader=DataLoader(train_dataset,batch_size=1,shuffle=True ,num_workers = 0)
test_dataset=dataset(test_data,test_lab,test_mask)
test_dataloader=DataLoader(test_dataset,batch_size=1,shuffle=True ,num_workers = 0)
total_dataset=dataset(total_data,total_lab,total_mask)
total_dataloader=DataLoader(total_dataset,batch_size=1,shuffle=True ,num_workers = 0)
