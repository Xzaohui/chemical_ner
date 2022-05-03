import imp
import os
import numpy as np
lab={'O': 0, 'ORG': 1,'MISC': 2, 'PER': 3, 'LOC': 4,'start':5,'end':6}
eng_train_dir = './dataset/eng.train'
eng_test_dir = './dataset/eng.test'
sent=[]
lab_index=[]
def get_train_data(dir):
    file_list = []
    temp_list=[]
    temp_list1=[]
    temp_list2=[]
    temp_list3=[]
    temp_list4=[]
    f=open(eng_train_dir,'r')
    str=f.read()
    str=str.split('\n')
    for i in str:
        temp_list.append(i.split(' '))
    for i in temp_list:
        if i[0]=='':
            file_list.append(temp_list2)
            sent.append(temp_list3)
            lab_index.append(temp_list4)
            temp_list2=[]
            temp_list3=[]
            temp_list4=[]
            continue
        temp_list1.append(i[0])
        temp_list1.append(i[3])
        temp_list2.append(temp_list1)
        temp_list3.append(i[0])
        temp_list4.append(lab[i[3]])
        temp_list1=[]
    return file_list
# print(get_train_data(eng_train_dir))
file_list=get_train_data(eng_train_dir)
lab_cont={}
word_cont={}
dic={}
i=0
for t_sent in file_list:
    for wd in t_sent:
        if wd[1] not in lab_cont:
            lab_cont[wd[1]]=1
        lab_cont[wd[1]]+=1
        if wd[0] not in word_cont:
            word_cont[wd[0]]=1
            dic[wd[0]]=i
            i+=1
        word_cont[wd[0]]+=1

sent_index=[]
for t_sent in sent:
    temp=[]
    for wd in t_sent:
        temp.append(dic[wd])
    sent_index.append(temp)
# print(lab_index[0])

#手动填充
# max_sent_len=0
# for t_sent in sent:
#     if len(t_sent)>max_sent_len:
#         max_sent_len=len(t_sent)
# lab_index_pad=lab_index
# sent_index_pad=[]
# for i in range(len(sent)):
#     temp=[]
#     for wd in sent[i]:
#         temp.append(dic[wd])
#     if len(sent[i])<max_sent_len:
#         temp.extend([0]*(max_sent_len-len(temp)))
#         lab_index_pad[i].extend([0]*(max_sent_len-len(sent[i])))
#     sent_index_pad.append(temp)



