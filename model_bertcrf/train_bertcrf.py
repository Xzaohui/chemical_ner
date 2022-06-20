import numpy as np
import model_bertcrf
import torch
import datetime
from transformers import BertTokenizer
import bert_data_manage
bert_path='C:/Users/83912/Desktop/project/chemical_ner/model/chemical-bert-uncased'
tokenizer_word=BertTokenizer.from_pretrained(bert_path)
model=model_bertcrf.model_bertcrf()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss=torch.nn.CrossEntropyLoss()

model.to("cuda")
model.train()
def len_sent(attention_mask):
    return torch.sum(attention_mask).item()

for j in range(5):
    for i,(train_data,attention_mask,train_lab) in enumerate(bert_data_manage.train_dataloader):
        optimizer.zero_grad()
        loss=model.loss(train_data,attention_mask,train_lab)
        loss.backward()
        optimizer.step()
        if i%10==0:
            print(datetime.datetime.now())
            print(loss)
            print("epoch:",j,"step:",i)
            print("=================================================")
        
model.eval()
# for i,(train_data,attention_mask,train_lab) in enumerate(bert_data_manage.train_dataloader):
    
#     print("==============================")
#     _,path=model(train_data,attention_mask)
#     print(len(path))
#     j=1
#     while(j<len(path)):
#         if path[j]==0:
#             m=tokenizer_word.convert_ids_to_tokens(train_data[0][j].tolist())
#             j+=1
#             while(j<len(path) and path[j]==1):
#                 m+=" "+tokenizer_word.convert_ids_to_tokens(train_data[0][j].tolist())
#                 j+=1
#             print("M "+m)
#             continue
#         if path[j]==2:
#             m=tokenizer_word.convert_ids_to_tokens(train_data[0][j].tolist())
#             j+=1
#             while(j<len(path) and path[j]==3):
#                 m+=" "+tokenizer_word.convert_ids_to_tokens(train_data[0][j].tolist())
#                 j+=1
#             print("R "+m)
#             continue
#         if path[j]==4:
#             m=tokenizer_word.convert_ids_to_tokens(train_data[0][j].tolist())
#             j+=1
#             while(j<len(path) and path[j]==5):
#                 m+=" "+tokenizer_word.convert_ids_to_tokens(train_data[0][j].tolist())
#                 j+=1
#             print("P "+m)
#             continue
#         if path[j]==6:
#             m=tokenizer_word.convert_ids_to_tokens(train_data[0][j].tolist())
#             j+=1
#             while(j<len(path) and path[j]==7):
#                 m+=" "+tokenizer_word.convert_ids_to_tokens(train_data[0][j].tolist())
#                 j+=1
#             print("M "+m)
#             continue
#         j+=1

cont_total=np.zeros((2,27))
cont_cur=np.zeros((1,27))
for i,(train_data,attention_mask,train_lab) in enumerate(bert_data_manage.train_dataloader):
        path_score,path_index=model(train_data,attention_mask)
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
    print("label:",list(bert_data_manage.BIO_lab.keys())[list(bert_data_manage.BIO_lab.values()).index(i)],"精确率:",precision)
    print("label:",list(bert_data_manage.BIO_lab.keys())[list(bert_data_manage.BIO_lab.values()).index(i)],"召回率:",recall)
    print("label:",list(bert_data_manage.BIO_lab.keys())[list(bert_data_manage.BIO_lab.values()).index(i)],"F1:",2*precision*recall/(precision+recall))
    print("-----------------------------------------")
    
