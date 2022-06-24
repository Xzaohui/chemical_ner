import random
import torch
import jsonlines
from torch.utils.data import Dataset, DataLoader
import data_manage
import datetime
from model_bilstmcrf import model_bilstmcrf
from model_lstmcrf import model_lstmcrf
data_path='C:/Users/83912/Desktop/project/chemical_ner/data/self_training.jsonl'
train_file = open(data_path, 'r', encoding='utf-8')
lstmcrf=model_lstmcrf.Lstm_model()
lstmcrf.load_state_dict(torch.load("C:/Users/83912/Desktop/project/chemical_ner/model/lstmcrf_negative.pkl"))
lstmcrf.to("cuda")
bilstmcrf=model_bilstmcrf.biLstm_model()
bilstmcrf.load_state_dict(torch.load("C:/Users/83912/Desktop/project/chemical_ner/model/bilstmcrf_negative.pkl"))
bilstmcrf.to("cuda")
def tokenization(data):
    t_data = data.replace(',', ' , ').replace('.', ' . ').replace('-', ' - ').replace('/', ' / ').replace(':', ' : ').replace(';', ' ; ').replace('@',' @ ').replace('(', ' ').replace(')', ' ').replace("%"," % ").split( ' ')  # abstract (ab) 需要变成 abstract ab 吗？？
    t_data = [x.strip() for x in t_data] # 去除单词首尾的空格
    data = list(filter(None, t_data))
    return data
class predict_dataset(Dataset):
    def __init__(self,total_datas,total_masks):
        self.total_data=total_datas
        self.total_mask=total_masks
    def __len__(self):
        return len(self.total_mask)
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

total_data=[]
for data in jsonlines.Reader(train_file):
    words=tokenization(data['data'])
    t_d=[]
    for i,word in enumerate(words):
        if word in data_manage.word2idx:
            t_d.append(data_manage.word2idx[word])
        else:
            t_d.append(random.randint(0,len(data_manage.w2v.wv.vocab)))
    total_data.append(t_d)

max_sent_len=600
total_mask=[]
delete_=[]
for i in range(len(total_data)):
    if(len(total_data[i])>max_sent_len):
        delete_.append(i)
        continue
    t_mask=[]
    t_mask.extend([1]*len(total_data[i]))
    t_mask.extend([0]*(max_sent_len-len(total_data[i])))
    total_data[i].extend([0]*(max_sent_len-len(total_data[i])))
    total_mask.append(t_mask)
for i in sorted(delete_,reverse=True):
    del total_data[i]
train_data=data_manage.total_data_pad
train_mask=data_manage.total_mask
train_lab=data_manage.total_lab_pad

total_data=torch.Tensor(total_data).cuda().long()
total_mask=torch.Tensor(total_mask).cuda().long()

predict=predict_dataset(total_data,total_mask)
predict_dataloader=DataLoader(predict,batch_size=1,shuffle=False)
train_len=142
train_data_co=train_data
train_mask_co=train_mask
train_lab_co=train_lab
for i,(data,mask) in enumerate(predict_dataloader):
    path_score_lstmcrf,path_lstmcrf=lstmcrf(data,mask)
    path_score_bilstmcrf,path_bilstmcrf=bilstmcrf(data,mask)
    path_score=max(path_score_lstmcrf,path_score_bilstmcrf)
    path=path_lstmcrf if path_score==path_score_lstmcrf else path_bilstmcrf
    print(path_score)
    if path_score>1500:
        train_data_co=torch.cat((train_data_co,data),0)
        train_mask_co=torch.cat((train_mask_co,mask),0)
        train_path=[]
        for j in range(len(path)):
            train_path.append(path[j].item())
        train_path.extend([data_manage.BIO_lab['E']]*(max_sent_len-len(train_path)))
        train_lab_co=torch.cat((train_lab_co,torch.Tensor(train_path).unsqueeze(0).to('cuda')),0)
    if len(train_data_co)-train_len>100:
        train_dataset=self_train_dataset(train_data_co,train_mask_co,train_lab_co)
        train_dataloader=DataLoader(train_dataset,batch_size=1,shuffle=True)
        for j,(data,mask,lab) in enumerate(train_dataloader):
            lstmcrf.train()
            bilstmcrf.train()
            optimizer1=torch.optim.Adam(lstmcrf.parameters(),lr=0.001)
            optimizer2=torch.optim.Adam(bilstmcrf.parameters(),lr=0.001)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss1=lstmcrf.loss(data,lab,mask)
            loss2=bilstmcrf.loss(data,lab,mask)
            loss1.backward()
            loss2.backward()
            optimizer1.step()
            optimizer2.step()
            if j%20==0:
                print(i)
                print(loss1.item())
                print(loss2.item())
                print(datetime.datetime.now())
                print("epoch:",train_len/100,"step:",j)
                print("=================================================")
        torch.save(lstmcrf.state_dict(), "C:/Users/83912/Desktop/project/chemical_ner/model/lstmcrf_"+datetime.datetime.now().strftime('%m-%d,%H_%M_%S')+".pkl")
        torch.save(bilstmcrf.state_dict(), "C:/Users/83912/Desktop/project/chemical_ner/model/bilstmcrf_"+datetime.datetime.now().strftime('%m-%d,%H_%M_%S')+".pkl")
        train_data_co=train_data
        train_mask_co=train_mask
        train_lab_co=train_lab