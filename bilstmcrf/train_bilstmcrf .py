from pickletools import optimize
import dataload
import model_bilstmcrf
import torch
import datetime

model=model_bilstmcrf.Lstm_model()
model.to("cuda")
optimize=torch.optim.Adam(model.parameters(),lr=0.001)


epoch=5
model.train()
for i in range(epoch):
    for j,(sent,lab) in enumerate(dataload.train_dataloader):
        optimize.zero_grad()
        loss=model.loss(sent,lab)
        loss=torch.mean(loss)
        loss.backward()
        optimize.step()
        if j%200==0:
            print(datetime.datetime.now())
            print(loss)
            print("epoch:",i,"step:",j)
            print("=================================================")
torch.save(model.state_dict(), "./model/bilstmcrf.pkl") 

