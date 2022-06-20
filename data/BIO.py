#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
"""
-----------------------------
# @FileName     :BIO.py
# @Time         :2022/6/10 10:53
# @Author       :Junbao
# @Software     :PyCharm Community Edition
-----------------------------
Description :
针对数据集构造训练BIO格式数据集，仅仅为数据处理，不包括切分训练集
"""
import jsonlines
class DataProcess:
    def __init__(self, input_file_path, out_file_path):
        self.papers_word2lablel = []
        error_num = 0

        file = open(input_file_path, 'r', encoding='utf-8')
        for item in jsonlines.Reader(file):
            sample_str = self.tokenization(item['data'])
            lab_list = [ [] for x in range(len(sample_str) )]
            # print('总词数',len(sample_str))

            item['label'] = sorted(item['label'], key=(lambda x: x[0]))     # 排序

            index = 0

            for item_i,item_label in  enumerate(item['label']):   # 创建等维度空列表，按 lab作为外循环
                begin_flag = True
                string = item['data'][ item_label[0]:item_label[1] ]
                sample = self.tokenization(string)
                item['label'][item_i].append(sample)
                for s in sample:
                    try:
                        if index == sample_str.index(s,index):
                            index = sample_str.index(s,index+1)     # 防止原地不动
                        else:
                            index = sample_str.index(s, index)
                        # print(item_i,item_label[0],item_label[1],item_label[2],'\t',index)
                        if begin_flag:  # !
                            lab_list[index] = 'B-' + item_label[2]
                            begin_flag = False
                        else:
                            lab_list[index] = 'I-' + item_label[2]
                    except ValueError :
                        print(item['id'],item['label'][item_i][:3],string,"\t",s)
                        # error_index = sample_str.index(s)
                        # print(item['id'],item_i,item_label[0],item_label[1],item_label[2],'\t',index,'\t','!!index!!!',error_index,sample_str[error_index])
                        # print()
                        error_num += 1

            merge = []
            for lab_i,lab in enumerate(lab_list):
                if lab == []:
                    lab_list[lab_i] = 'O'
                merge.append(lab_list[lab_i]  )

            self.papers_word2lablel.append(merge)
        print('错误总数： ',error_num)
        print('***** congratulations ! *****')

    def tokenization(self,data):

        c_ = data.replace(',', ' , ').replace('.', ' . ').replace('-', ' - ').replace('/', ' / ').replace(':', ' : ').replace(';', ' ; ').replace('@',' @ ').replace('(', ' ').replace(')', ' ').replace("%"," % ").split( ' ')  # abstract (ab) 需要变成 abstract ab 吗？？
        c = [x.strip() for x in c_] # 去除单词首尾的空格
        c = list(filter(None, c))
        return c

def save_txt(word_cont):
    # 由于文件中有多行，直接读取会出现错误，因此一行一行读取
    file = open('./savelabels'+'.txt', 'w', encoding='utf-8')
    for x in word_cont:
        file.write(str(x))
        # file.write(str(v))
        file.write('\n')
    print('finsh saving')

if __name__ == '__main__':
    raw_data ='C:/Users/83912/Desktop/project/chemical_ner/data/4-our_last.jsonl'
    out_data_path = "./"
    # 会生成train.txt及label.txt
    dataProcess = DataProcess(raw_data, out_data_path)
    save_txt(dataProcess.papers_word2lablel)
    print()



