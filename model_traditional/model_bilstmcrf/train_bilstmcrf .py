import model_bilstmcrf
import torch
import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import data_manage

model=model_bilstmcrf.biLstm_model()
model.to("cuda")
optimize=torch.optim.AdamW(model.parameters(),lr=0.01, eps=1e-8)


epoch=5
def train(model,epoch,dataloader):
    model.train()
    for i in range(epoch):
        for j,(data,lab,mask) in enumerate(dataloader):
            optimize.zero_grad()
            loss=model.loss(data,lab,mask)
            loss=torch.mean(loss)
            loss.backward()
            optimize.step()
            if j%10==0:
                print(datetime.datetime.now())
                print(loss)
                print("epoch:",i,"step:",j)
                print("=================================================")
    torch.save(model.state_dict(), "C:/Users/83912/Desktop/project/chemical_ner/model/bilstmcrf_"+datetime.datetime.now().strftime('%m-%d,%H_%M_%S')+".pkl")
