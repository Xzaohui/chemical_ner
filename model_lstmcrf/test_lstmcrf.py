import torch
import data_manage
import model_lstmcrf
import numpy as np
model=model_lstmcrf.Lstm_model()
model.to("cuda")
model.load_state_dict(torch.load("./model/lstmcrf.pkl"))
cont_total=np.zeros((2,27))
cont_cur=np.zeros((1,27))
def len_sent(attention_mask):
    return torch.sum(attention_mask).item()
model.eval()

def predict():
    for i,(train_data,train_lab,attention_mask) in enumerate(data_manage.train_dataloader):
        
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

def p_r_f():
    for i,(train_data,train_lab,attention_mask) in enumerate(data_manage.train_dataloader):
        path_score,path_index=model(train_data,attention_mask)
        # lab=lab[0].cpu().numpy()
        # path_index.cpu()
        # path=torch.Tensor([])
        # for tpath in path_index:
        #     path=torch.cat((path,tpath.unsqueeze(0)),0)
        # path=path.int().numpy()
        train_lab=train_lab[0].cpu().numpy()
        for j in range(len(path_index)):
            cont_total[0][train_lab[j]]+=1
            cont_total[1][path_index[j]]+=1
            if train_lab[j]==path_index[j]:
                cont_cur[0][path_index[j]]+=1
    i=0

    for i in range(25):
        precision=cont_cur[0][i]/cont_total[1][i]
        recall=cont_cur[0][i]/cont_total[0][i]
        print("label:",list(data_manage.BIO_lab.keys())[list(data_manage.BIO_lab.values()).index(i)],"精确率:",precision)
        print("label:",list(data_manage.BIO_lab.keys())[list(data_manage.BIO_lab.values()).index(i)],"召回率:",recall)
        print("label:",list(data_manage.BIO_lab.keys())[list(data_manage.BIO_lab.values()).index(i)],"F1:",2*precision*recall/(precision+recall))
        print("-----------------------------------------")
