import torch
import model_bilstmcrf
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import data_manage
model=model_bilstmcrf.biLstm_model()
model.to("cuda")
model.load_state_dict(torch.load("C:/Users/83912/Desktop/project/chemical_ner/model/bilstmcrf_06-23,15_29_22.pkl"))
cont_total=np.zeros((2,len(data_manage.BIO_lab)))
cont_cur=np.zeros((1,len(data_manage.BIO_lab)))
def len_sent(attention_mask):
    return torch.sum(attention_mask).item()
model.eval()

def predict():
    for i,(train_data,train_lab,attention_mask) in enumerate(data_manage.test_dataloader):
        
        print("==============================")
        _,path=model(train_data,attention_mask)
        print(len(path))
        j=1
        while(j<len(path)):
            if path[j]==0:
                m=data_manage.idx2word[train_data[0][j].item()]
                j+=1
                while(j<len(path) and path[j]==1):
                    m+=" "+data_manage.idx2word[train_data[0][j].item()]
                    j+=1
                print("M "+m)
                continue
            if path[j]==2:
                m=data_manage.idx2word[train_data[0][j].item()]
                j+=1
                while(j<len(path) and path[j]==3):
                    m+=" "+data_manage.idx2word[train_data[0][j].item()]
                    j+=1
                print("R "+m)
                continue
            if path[j]==4:
                m=data_manage.idx2word[train_data[0][j].item()]
                j+=1
                while(j<len(path) and path[j]==5):
                    m+=" "+data_manage.idx2word[train_data[0][j].item()]
                    j+=1
                print("P "+m)
                continue
            if path[j]==6:
                m=data_manage.idx2word[train_data[0][j].item()]
                j+=1
                while(j<len(path) and path[j]==7):
                    m+=" "+data_manage.idx2word[train_data[0][j].item()]
                    j+=1
                print("M "+m)
                continue
            j+=1

def test_average_score():
    with torch.no_grad():
        score=torch.zeros(1).to('cuda')
        i=0
        for i,(test_data,test_lab,attention_mask) in enumerate(data_manage.test_dataloader):
            path_score,path_index=model(test_data,attention_mask)
            score+=path_score
        print(score/i)

def train_average_score():
    with torch.no_grad():
        score=torch.zeros(1).to('cuda')
        i=0
        for i,(test_data,test_lab,attention_mask) in enumerate(data_manage.train_dataloader):
            path_score,path_index=model(test_data,attention_mask)
            score+=path_score
        print(score/i)

def p_r_f():
    for i,(train_data,train_lab,attention_mask) in enumerate(data_manage.test_dataloader):
        path_score,path_index=model(train_data,attention_mask)
        train_lab=train_lab[0].cpu().numpy()
        for j in range(len(path_index)):
            cont_total[0][train_lab[j]]+=1
            cont_total[1][path_index[j]]+=1
            if train_lab[j]==path_index[j]:
                cont_cur[0][path_index[j]]+=1
    i=0

    for i in range(len(data_manage.BIO_lab)):
        precision=cont_cur[0][i]/cont_total[1][i]
        recall=cont_cur[0][i]/cont_total[0][i]
        print("label:",list(data_manage.BIO_lab.keys())[list(data_manage.BIO_lab.values()).index(i)],"精确率:",precision)
        print("label:",list(data_manage.BIO_lab.keys())[list(data_manage.BIO_lab.values()).index(i)],"召回率:",recall)
        print("label:",list(data_manage.BIO_lab.keys())[list(data_manage.BIO_lab.values()).index(i)],"F1:",2*precision*recall/(precision+recall))
        print("-----------------------------------------")
    print("总的精确率:",cont_cur[0].sum()/cont_total[1].sum())
    print("总的召回率:",cont_cur[0].sum()/cont_total[0].sum())
    print("总的F1:",2*cont_cur[0].sum()/(cont_total[1].sum()+cont_total[0].sum()))
    print("=================================================")


p_r_f()
train_average_score()
test_average_score()