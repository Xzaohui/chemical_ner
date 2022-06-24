import torch
import json
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
import model_bertcrf
import train_bertcrf
import test_bertcrf
import datetime
data_path='C:/Users/83912/Desktop/project/chemical_ner/data/self_training.jsonl'
bert_path='C:/Users/83912/Desktop/project/chemical_ner/model/chemical-bert-uncased'
tokenizer = BertTokenizerFast.from_pretrained(bert_path)


class predict_dataset(Dataset):
    def __init__(self,total_data,total_mask):
        self.total_data=total_data
        self.total_mask=total_mask
    def __len__(self):
        return len(self.total_data)
    def __getitem__(self,idx):
        return self.total_data[idx],self.total_mask[idx]

class self_train_dataset(Dataset):
    def __init__(self,total_data,total_mask,total_lab):
        self.total_data=total_data
        self.total_mask=total_mask
        self.total_lab=total_lab
    def __len__(self):
        return len(self.total_data)
    def __getitem__(self,idx):
        return self.total_data[idx],self.total_mask[idx],self.total_lab[idx]

def read_json(path):
    file = open(path, 'r', encoding='utf-8')
    papers = []
    for line in file.readlines():
        dic = json.loads(line)
        papers.append(dic)
    file.close()
    return papers

train_json=read_json(data_path)
t_data=[]
for data in train_json:
    t_data.append(data['data'])

t=tokenizer(t_data,truncation=True,padding='max_length',max_length=512,return_tensors="pt")
total_data=t['input_ids']
total_mask=t['attention_mask']
total_data=total_data.cuda().long()
total_mask=total_mask.cuda().long()

predict_dataset=predict_dataset(total_data,total_mask)
predict_dataloader=DataLoader(predict_dataset,batch_size=1,shuffle=False)
self_train_data=[]
self_train_mask=[]
self_train_lab=[]

imp=0.2119815668202765
for i,(train_data,attention_mask) in enumerate(predict_dataloader):
    model=model_bertcrf.model_bertcrf()
    model.load_state_dict(torch.load("C:/Users/83912/Desktop/project/chemical_ner/model/bertcrf.pkl"))
    model.eval()
    model.to("cuda")
    with torch.no_grad():
        path_score,path_index=model(train_data,attention_mask)
        print(path_score)
        if path_score>3000:
            self_train_data.append(train_data[0])
            self_train_mask.append(attention_mask[0])
            path=[]
            for path_i in path_index:
                path.append(path_i.item())
            path.extend([26]*(512-len(path)))
            path=torch.tensor(path).cuda().long()
            self_train_lab.append(path)
    
    if len(self_train_data)>100:
        self_train_dataset=self_train_dataset(self_train_data,self_train_mask,self_train_lab)
        self_train_dataloader=DataLoader(self_train_dataset,batch_size=1,shuffle=False)
        train_bertcrf.train(model,self_train_dataloader)
        if test_bertcrf.imp_p_r_f(model)>imp:
            torch.save(model.state_dict(), "C:/Users/83912/Desktop/project/chemical_ner/model/bertcrf.pkl")
            model.save_bert(bert_path)
            print("=================saved model===========================")
            print(datetime.datetime.now())
        self_train_data=[]
        self_train_mask=[]
        self_train_lab=[]