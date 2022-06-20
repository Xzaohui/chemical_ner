import data_manage
import model_bilstmcrf
import torch
import datetime

model=model_bilstmcrf.biLstm_model()
model.to("cuda")
optimize=torch.optim.Adam(model.parameters(),lr=0.01)


epoch=5
model.train()
for i in range(epoch):
    for j,(data,lab,mask) in enumerate(data_manage.total_dataloader):
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
torch.save(model.state_dict(), "./model/bilstmcrf.pkl") 

