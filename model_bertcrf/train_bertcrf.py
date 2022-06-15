from regex import P
import model_bertcrf
import torch
import datetime
import bert_dataload
from transformers import BertTokenizer
import bert_predata

bert_path='./model/chemical-bert-uncased'
tokenizer_word=BertTokenizer.from_pretrained(bert_path)
# model=model_bertcrf.model
model=model_bertcrf.model_bertcrf()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss=torch.nn.CrossEntropyLoss()

model.to("cuda")
model.train()

for j in range(5):
    for i,(train_data,attention_mask,train_lab) in enumerate(bert_dataload.train_dataloader):
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
for i,(train_data,attention_mask,train_lab) in enumerate(bert_dataload.train_dataloader):
    _,path=model(train_data,attention_mask)
    j=1
    print("==============================")
    word=tokenizer_word.tokenize(bert_predata.t_data[i])
    while(j-1<len(word)):
        if path[j]==2:
            m=word[j-1]
            j+=1
            while(j-1<len(word) and path[j]==3):
                m+=" "+word[j-1]
                j+=1
            print(m)
            continue
        j+=1
    
