import random
import jsonlines
import functools
from gensim.models import word2vec
from torch.utils.data import Dataset, DataLoader
import torch
lab={'M':0,'R':1,'P':2,'F':3,'ELE':4,'CA':5,'CO2':6,'RE':7,'BM':8,'BR':9,'BP':10,'BF':11,'O':12}
# BIO_lab={'B-M':0,'I-M':1,'B-R':2,'I-R':3,'B-P':4,'I-P':5,'B-F':6,'I-F':7,'B-ELE':8,'I-ELE':9,'B-CA':10,'I-CA':11,'B-CO2':12,'I-CO2':13,'B-RE':14,'I-RE':15,'B-BM':16,'B-BR':18,'B-BP':20,'B-BF':22,'I-BM':17,'I-BR':19,'I-BP':21,'I-BF':23,'O':24,'S':25,'E':26}
BIO_lab={'B-M':0,'I-M':1,'B-R':2,'I-R':3,'B-P':4,'I-P':5,'B-F':6,'I-F':7,'B-ELE':8,'I-ELE':9,'B-CA':10,'I-CA':11,'B-CO2':12,'I-CO2':13,'B-RE':14,'I-RE':15,'O':16,'S':17,'E':18}
# train_file = open('C:/Users/83912/Desktop/project/chemical_ner/data/5-our_last.jsonl', 'r', encoding='utf-8')
train_file = open('C:/Users/83912/Desktop/project/chemical_ner/data/5-noB_our_last.jsonl', 'r', encoding='utf-8')
w2v=word2vec.Word2Vec.load('C:/Users/83912/Desktop/project/chemical_ner/model/chemical256.w2v')
word2idx = {word:i for i, word in enumerate(w2v.wv.index2word)}
idx2word = {i:word for i, word in enumerate(w2v.wv.index2word)}
def tokenization(data):
    t_data = data.replace(',', ' , ').replace('.', ' . ').replace('-', ' - ').replace('/', ' / ').replace(':', ' : ').replace(';', ' ; ').replace('@',' @ ').replace('(', ' ').replace(')', ' ').replace("%"," % ").split( ' ')  # abstract (ab) 需要变成 abstract ab 吗？？
    t_data = [x.strip() for x in t_data] # 去除单词首尾的空格
    data = list(filter(None, t_data))
    return data

def cmp(x,y):
    if x[0]>y[0]:
        return 1
    elif x[0]<y[0]:
        return -1
    return 0

def is_iof(s,i):
    try:
        float(s[i])
    except ValueError:
        return 0
    if i<len(s)-2 and s[i+1]=='.':
        try:
            float(s[i+2])
            return 2
        except ValueError:
            return 1
    else:
        return 1

total_data=[]
train_words=[]
total_lab=[]
t_lab=[]
for data in jsonlines.Reader(train_file):
    words=tokenization(data['data'])
    t_d=[]
    for i,word in enumerate(words):
        if word in word2idx:
            t_d.append(word2idx[word])
        else:
            t_d.append(random.randint(0,len(w2v.wv.vocab)))
    total_data.append(t_d)
    train_words.append(words)
    t_l=data['label']
    t_l.sort(key=functools.cmp_to_key(cmp))
    for i in range(len(t_l)):
        t_l[i].append(tokenization(data['data'][t_l[i][0]:t_l[i][1]]))
    t_lab.append(t_l)
train_file.close()
for i in range(len(train_words)):
    num_lab=0
    lab_index=[]
    j=0
    while j<len(train_words[i]):
        if num_lab<len(t_lab[i]) and train_words[i][j]==t_lab[i][num_lab][3][0]:
            lab_index.append(lab[t_lab[i][num_lab][2]]*2)
            j+=1
            if len(t_lab[i][num_lab][3])>1:
                k=1
                while k<len(t_lab[i][num_lab][3]):
                    lab_index.append(lab[t_lab[i][num_lab][2]]*2+1)
                    j+=1
                    k+=1
            num_lab+=1
        else:
            lab_index.append(BIO_lab['O'])
            j+=1
    total_lab.append(lab_index)

# 手动填充
max_sent_len=600
total_lab_pad=total_lab
total_data_pad=total_data
total_mask=[]
for i in range(len(total_lab)):
    t_mask=[]
    t_mask.extend([1]*len(total_data[i]))
    t_mask.extend([0]*(max_sent_len-len(total_data[i])))
    total_lab_pad[i].extend([BIO_lab['E']]*(max_sent_len-len(total_lab[i])))
    total_data_pad[i].extend([0]*(max_sent_len-len(total_data[i])))
    total_mask.append(t_mask)

class data_loader(Dataset):
    def __init__(self,total_data,total_lab,total_mask):
        self.total_data=total_data
        self.total_lab=total_lab
        self.total_mask=total_mask
    def __len__(self):
        return len(self.total_data)
    def __getitem__(self,idx):
        return self.total_data[idx],self.total_lab[idx],self.total_mask[idx]

# #自动填充
# sent=pad_sequence([torch.LongTensor(i) for i in pre_data.sent_index], batch_first=True, padding_value=0)
# lab=pad_sequence([torch.LongTensor(i) for i in pre_data.lab_index], batch_first=True, padding_value=0)

total_lab_pad=torch.Tensor(total_lab_pad)
total_data_pad=torch.Tensor(total_data_pad)
total_mask=torch.Tensor(total_mask)

total_lab_pad=total_lab_pad.cuda().long()
total_data_pad=total_data_pad.cuda().long()
total_mask=total_mask.cuda().long()

# total_dataset=data_loader(total_data_pad,total_lab_pad,total_mask)
# total_dataloader=DataLoader(total_dataset,batch_size=1,shuffle=True ,num_workers = 0)

train_data=total_data_pad[:int(len(total_data_pad)*0.9)]
train_lab=total_lab_pad[:int(len(total_data_pad)*0.9)]
train_mask=total_mask[:int(len(total_data_pad)*0.9)]
test_data=total_data_pad[int(len(total_data_pad)*0.9):]
test_lab=total_lab_pad[int(len(total_data_pad)*0.9):]
test_mask=total_mask[int(len(total_data_pad)*0.9):]

# for lab in train_lab:
#     for i in range(len(lab)):
#         if lab[i]==BIO_lab['O']:
#             if  random.random()>0.97:
#                 lab[i]=random.randint(0,len(BIO_lab)-3)

train_dataset=data_loader(train_data,train_lab,train_mask)
train_dataloader=DataLoader(train_dataset,batch_size=1,shuffle=True ,num_workers = 0)
test_dataset=data_loader(test_data,test_lab,test_mask)
test_dataloader=DataLoader(test_dataset,batch_size=1,shuffle=True ,num_workers = 0)