#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
-----------------------------
# @FileName     :count_content.py
# @Time         :2022/5/17 20:23
# @Author       :Junbao
# @Software     :PyCharm Community Edition
-----------------------------
Description :

"""
import json


def read_jsonl(path):
    # 由于文件中有多行，直接读取会出现错误，因此一行一行读取
    file = open(path, 'r', encoding='utf-8')
    papers = []
    for line in file.readlines():
        dic = json.loads(line)
        papers.append(dic)
    print('数据样本总数为：',len(papers))
    return papers

def papers_count(list_str_):
    list_str = [[x.replace(",", "").replace(".", "").replace("\nDoi:","").replace("Title:", "").replace("\nAbstract:", " ")] for x in list_str_]
    word_cont={}
    dic={}
    i=0
    for t_sent in list_str:
        for wd in t_sent[0].split(' '):
            if wd not in word_cont:
                word_cont[wd]=1
                dic[wd]=i            # 从词到数字的映射；该映射的数量 与word_cont的种类一致
                i+=1
            word_cont[wd]+=1         # 统计：各个词的数量 ； 一共23623种词
    print('word_cont_num : ',i)
    return word_cont,dic

def save_txt(word_cont,num):

    file = open('./our_paper_conunt'+str(num)+'.txt', 'w', encoding='utf-8')
    for x in word_cont:
        file.write(str(x))
        # file.write(str(v))
        file.write('\n')
    print('finsh saving')


if __name__ == "__main__":
    # path = r'D:\science\BJ_BIG\赛题四-样本数据\样本数据.jsonl'
    path = r'D:\science\BJ_BIG\output.jsonl'
    papers = read_jsonl(path)
    paper_datas = []

    for x in papers:
        paper_datas.append(x['data'])
    word_cont_,dic = papers_count(paper_datas)
    word_cont =  sorted(word_cont_.items(),key=lambda x:x[1],reverse=True)
    save_txt(word_cont,120)
    print('congratulations!')
