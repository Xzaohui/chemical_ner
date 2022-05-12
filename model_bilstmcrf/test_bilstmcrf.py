from pandas import array
import torch
import dataload
import model_bilstmcrf
import numpy as np
import pre_data
model=model_bilstmcrf.Lstm_model()
model.to("cuda")
model.load_state_dict(torch.load("./model/bilstmcrf.pkl"))
cont_total=np.zeros((2,5))
cont_cur=np.zeros((1,5))
for i,(sent,lab) in enumerate(dataload.test_dataloader):
    path_score,path_index=model(sent)
    lab=lab[0].cpu().numpy()
    path=torch.Tensor([])
    for tpath in path_index:
        tpath=tpath.cpu()
        path=torch.cat((path,tpath.unsqueeze(0)),0)
    path=path.int().numpy()
    for j in range(len(path)):
        cont_total[0][lab[j]]+=1
        cont_total[1][path[j]]+=1
        if lab[j]==path[j]:
            cont_cur[0][lab[j]]+=1
i=0

for i in range(5):
    precision=cont_cur[0][i]/cont_total[1][i]
    recall=cont_cur[0][i]/cont_total[0][i]
    print("label:",list(pre_data.lab.keys())[list(pre_data.lab.values()).index(i)],"精确率:",precision)
    print("label:",list(pre_data.lab.keys())[list(pre_data.lab.values()).index(i)],"召回率:",recall)
    print("label:",list(pre_data.lab.keys())[list(pre_data.lab.values()).index(i)],"F1:",2*precision*recall/(precision+recall))
    print("-----------------------------------------")