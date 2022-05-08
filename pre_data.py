import imp
import os
import numpy as np
lab={'o': 0, 'org': 1,'misc': 2, 'per': 3, 'loc': 4,'start':5,'end':6}
eng_train_dir = './dataset/eng.train'
eng_test_dir = './dataset/eng.testb'
sent=[]
sent1=[]
lab_train=[]
sent_train=[]
lab_test=[]
sent_test=[]
def get_train_data(dir):
    file_list = []
    temp_list=[]
    temp_list1=[]
    temp_list2=[]
    temp_list3=[]
    temp_list4=[]
    f=open(dir,'r')
    str=f.read()
    str=str.split('\n')
    for i in str:
        temp_list.append(i.split(' '))
    for i in temp_list:
        if i[0]=='':
            file_list.append(temp_list2)
            sent.append(temp_list3)
            lab_train.append(temp_list4)
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

train_file_list=get_train_data(eng_train_dir)
lab_cont={}
word_cont={}
dic={}
i=0
for t_sent in train_file_list:
    for wd in t_sent:
        if wd[1] not in lab_cont:
            lab_cont[wd[1]]=1
        lab_cont[wd[1]]+=1
        if wd[0] not in word_cont:
            word_cont[wd[0]]=1
            dic[wd[0]]=i
            i+=1
        word_cont[wd[0]]+=1


for t_sent in sent:
    temp=[]
    for wd in t_sent:
        temp.append(dic[wd])
    sent_train.append(temp)
# print(lab_index[0])

def get_test_data(dir):
    file_list = []
    temp_list=[]
    temp_list1=[]
    temp_list2=[]
    temp_list3=[]
    temp_list4=[]
    f=open(dir,'r')
    str=f.read()
    str=str.split('\n')
    for i in str:
        temp_list.append(i.split(' '))
    for i in temp_list:
        if i[0]=='':
            file_list.append(temp_list2)
            sent1.append(temp_list3)
            lab_test.append(temp_list4)
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
get_test_data(eng_test_dir)
for t_sent in sent1:
    temp=[]
    for wd in t_sent:
        if wd in dic:
            temp.append(dic[wd])
        else:
            temp.append(dic['0'])
    sent_test.append(temp)


# 手动填充
max_sent_len=130
lab_train_pad=lab_train
sent_train_pad=sent_train
for i in range(len(sent)):
    lab_train_pad[i].extend([0]*(max_sent_len-len(sent[i])))
    sent_train_pad[i].extend([0]*(max_sent_len-len(sent[i])))

lab_test_pad=lab_test
sent_test_pad=sent_test
i=0
for i in range(len(sent1)):
    lab_test_pad[i].extend([0]*(max_sent_len-len(sent1[i])))
    sent_test_pad[i].extend([0]*(max_sent_len-len(sent1[i])))

