#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
"""
-----------------------------
# @FileName     :4-slabels2blabels.py
# @Time         :2022/5/25 18:04
# @Author       :Junbao
# @Software     :PyCharm Community Edition
-----------------------------
Description :
    小标签转变为8个大标签；同时，自动标注额外的大标签
"""

import json
import jsonlines

def read_jsonl(path):
    # 由于文件中有多行，直接读取会出现错误，因此一行一行读取
    file = open(path, 'r', encoding='utf-8')
    papers = []
    for line in file.readlines():
        dic = json.loads(line)
        papers.append(dic)
    print()
    print('数据样本总数为：',len(papers))
    return papers

def add_extra_labels(papers):
    word2labels = {}

    # 添加额外的大标签
    word2labels['co2'] = 'CO2'
    word2labels['carbon dioxide'] = 'CO2'
    word2labels['reduction'] = 'RE'
    ELE = [ 'electrochemical','electrodes','electrode','electroreduction',
            'electrocatalysts','electrocatalytic','electrocatalyst' ,
            'electrolyte','electrolysis','electron','electrodeposition',
            'electrodeposited','electrocatalysis','electro-reduction','electrocatalytically' ]
    for x in ELE:
        word2labels[x] = 'ELE'
    CA =  ['catalytic','catalysts','catalyst','catalyzed']
    for x in CA:
        word2labels[x] = 'CA'

    papers_tag_blank = [
        [x['data'].replace(",", " ").replace(".", " ").replace("?", " ").replace(";", " ").replace(":", " ")] for x in
        papers]
    save_paper_label_list = []

    for papers_blank_ in papers_tag_blank:
        tagging_label = []
        for k, v in word2labels.items():
            idx_l = 0
            idx_l = papers_blank_[0].find(' ' + k + ' ', idx_l) + 1  # 因为这位置是空格，所以加一
            while idx_l != 0:  # find 只返回第一个
                idx_r = idx_l + len(k)
                # print(idx_l,idx_r,papers_blank_[0][idx_l-1:idx_r+1],'\t',v)
                tagging_label.append([idx_l+2, idx_r+2, v])         # 开头的引号在这儿不记录，但是记事本加入了
                idx_l = papers_blank_[0].find(' ' + k + ' ', idx_r) + 1  # 因为这位置是空格，所以加一
        # save_paper_data_list.append(papers_blank_)
        save_paper_label_list.append(tagging_label)
    print('****finshing tagging!****')

    for i in range(len(papers)):
        papers[i]['label'] = save_paper_label_list[i] + papers[i]['label']

    return papers



def dic2jsonl_label(papers,save_name,save_num):

    epochs = len(papers)
    with jsonlines.open(save_name + '_last'  + '.jsonl',mode='w') as writer:
        for idx in range(epochs):
            paper ={}
            # 更换成标签
            for i in range(len(papers[idx]['label'])-1,-1,-1):     # 倒叙 del 不会出现问题
                l = papers[idx]['label'][i]
                try :
                    if l[2][0] == 'B' :
                        new_label = 'B' + slabel2blabel(l[2][1:])
                    else :
                        new_label = slabel2blabel(l[2])
                    papers[idx]['label'][i][2] = new_label

                except: # 单个情况 如 [P]  pass
                    if len(l)==1:
                        print(l)
                        del papers[idx]['label'][i]
                        # pass
                    else:
                        print(idx,'\t',str(i),l)
                        papers[idx]['label'][i] = ["???"]

            # 标签检查与删除
            ls = sorted(papers[idx]['label'],key=(lambda x:x[0]))
            for i in range(len(ls) -1 ,0,-1):  # 倒序
                del_list = []
                if not ls[i][0] > ls[i-1][1] :   # 不合理
                    l = (ls[i-1][1]-ls[i-1][0])
                    r = (ls[i][1]-ls[i][0])
                    if  r > l  : # 谁大删谁
                        print(l,'\t',ls[i-1],'\t\t',r,'\t',ls[i],'\t',"删去 后面")
                        del_list.append(i)
                    else:
                        print(l,'\t',ls[i-1],'\t\t',r,'\t',ls[i],'\t',"删去 前面")
                        del_list.append(i-1)
                del_list = list(set(del_list))
                while len(del_list) >0:
                    i_del = del_list.pop(0)
                    print('del ',ls[i_del])
                    del ls[i_del]

            paper['id'] = save_num
            paper['data'] = 'title:' +  papers[idx]['data'][4:]
            paper['label'] = ls
            writer.write(paper)
            save_num +=1

    print('The total number of papers to be marked is:  ',save_num)

def slabel2blabel(label):
    F = ['Faradaic efficiency']

    M = ['CuMOx', 'CuOx', 'Cu-M', 'Cu/C', 'Cu-MOx', 'Cu molecular complex', 'Cu-MOF', 'CuSx', 'CuMSx', 'CuNx', 'Cu-LDH',
         'CuPx', 'Cu']

    R = ['structure control', 'surface modification', 'alloy', 'atomic level dispersion', 'composite', 'defect']

    P = ['HCOOH', 'CO', 'CH4', 'C2+', 'C2H4', 'C2H5OH', 'C2H6', 'acetone', 'HCHO', 'propanol', 'syngas','CH3OH']

    if label in M:
        return 'M'
    elif label in R:
        return 'R'
    elif label in P:
        return 'P'
    elif label in F:
        return 'F'
    else:
        return label

    # BM = []
    # BR = []
    # BP = []
    # BF = []
    # for m in M:
    #     BM.append('B'+ m)
    # for f in F:
    #     BM.append('B'+ f)
    # for r in R:
    #     BM.append('B'+ r)
    # for p in P:
    #     BM.append('B'+ p)


    # return F,M,R,P,BR,BF,BP,BM

if __name__ == "__main__":

    # 由于文件中有多行，直接读取会出现错误，因此一行一行读取
    # 我们的手标 文件;  our_manul.jsonl
    # 组委会手标文件 ：'样本数据 (2)_add.jsonl'
    file = open(r'our_manul.jsonl', 'r', encoding='utf-8')
    save_name = r'./4-our'
    save_num = 9000

    # file = open(r'样本数据 (2)_add.jsonl', 'r', encoding='utf-8')
    # save_name = r'./4-official'
    # save_num = 1000


    papers = []
    for line in file.readlines():
        try:
            dic = json.loads(line)
        except:
            print(line)
        papers.append(dic)

    papers_save = []
    for i in range(int(len(papers) / 2)):
        p = {}
        p['data'] = papers[2 * i]
        p['label'] = papers[2 * i + 1]
        papers_save.append(p)

    # 加上大标签

    papers_extra_labes = add_extra_labels(papers_save)

    # choose_manual = 5
    # dic2jsonl(papers_save,choose_manual,save_name=r'./manual'  )
    dic2jsonl_label(papers_extra_labes,save_name=save_name ,save_num = save_num )



    print(len(papers))

