from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
import torch
import bert_data_manage
import datetime

model=BertForTokenClassification.from_pretrained(bert_data_manage.bert_path,num_labels=len(bert_data_manage.BIO_lab)) 
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(bert_data_manage.train_dataloader) * 1
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=total_steps)
model.to('cuda')
model.train()
for j in range(2):
    for i,(train_data,train_lab,attention_mask) in enumerate(bert_data_manage.total_dataloader):
        out=model(train_data,attention_mask=attention_mask,labels=train_lab)
        loss=out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if i%10==0:
            print(datetime.datetime.now())
            print(loss)
            print("epoch:",j,"step:",i)
            print("=================================================")

model.save_pretrained(bert_data_manage.bert_path)

