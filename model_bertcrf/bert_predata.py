import torch
import json
import numpy as np
import transformers
from transformers import BertTokenizerFast
import functools

lab={'M':0,'R':1,'P':2,'F':3,'ELE':4,'CA':5,'CO2':6,'RE':7,'BM':8,'BR':9,'BP':10,'BF':11,'O':12}
BIO_lab={'B-M':0,'I-M':1,'B-R':2,'I-R':3,'B-P':4,'I-P':5,'B-F':6,'I-F':7,'B-ELE':8,'I-ELE':9,'B-CA':10,'I-CA':11,'B-CO2':12,'I-CO2':13,'B-RE':14,'I-RE':15,'B-BM':16,'B-BR':18,'B-BP':20,'B-BF':22,'I-BM':17,'I-BR':19,'I-BP':21,'I-BF':23,'O':24,'S':25,'E':26}

data_path='./data/'
bert_path='./model/chemical-bert-uncased'
tokenizer = BertTokenizerFast.from_pretrained(bert_path)

train_data_path=data_path+'4-our_last.jsonl'

def read_json(path):
    file = open(path, 'r', encoding='utf-8')
    papers = []
    for line in file.readlines():
        dic = json.loads(line)
        papers.append(dic)
    return papers

def cmp(x,y):
    if x[0]>y[0]:
        return 1
    elif x[0]<y[0]:
        return -1
    return 0

train_json=read_json(train_data_path)
t_data=[]
for data in train_json:
    t_data.append(data['data'])
t_lab=[]
for data in train_json:
    t_l=data['label']
    t_l.sort(key=functools.cmp_to_key(cmp))
    t_lab.append(t_l)


t=tokenizer(t_data,truncation=True,padding=True,return_offsets_mapping=True,max_length=600,return_tensors="pt")
train_data=t['input_ids']
attention_mask=t['attention_mask']
offset_mapping=t['offset_mapping'].numpy()

train_lab=[]
sent_idx=0
for sent_mapping in offset_mapping:
    t_l=[]
    word_idx=0
    i=1
    t_l.append(BIO_lab['S'])
    while(i<len(sent_mapping)):
        if sent_mapping[i][0]==0 and sent_mapping[i][1]==0:
            t_l.append(26)
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
            t_l.append(24)
            i+=1
    train_lab.append(t_l)
    sent_idx+=1
train_lab=torch.tensor(train_lab)