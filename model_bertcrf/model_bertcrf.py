from transformers import BertForTokenClassification
import torch
import torch.nn as nn

bert_path='./model/chemical-bert-uncased'
# model=BertForTokenClassification.from_pretrained(bert_path,num_labels=25)
def argmax(vec,axis):
    # return the argmax as a python int
    _, idx = torch.max(vec, axis)
    return idx
def max(vec,axis):
    # return the argmax as a python int
    max, _ = torch.max(vec, axis)
    return max
class model_bertcrf(nn.Module):
    def __init__(self):
        super(model_bertcrf,self).__init__()
        self.bert=BertForTokenClassification.from_pretrained(bert_path,num_labels=27)
        self.transitions=nn.Parameter(torch.empty(27,27))
        nn.init.uniform_(self.transitions, 0, 1) 
        self.transitions.data[:,25]=-1000
        self.transitions.data[26,:]=-1000
        self.transitions.data[25,26]=-1000
    def likely_matrix(self,train_data,attention_mask):
        out=self.bert(train_data,attention_mask=attention_mask)
        return out.logits
    def score_sentence(self,lab,l_matrix,attention_mask):
        score=l_matrix[torch.arange(1,dtype=torch.long),0,lab[:,0]] #第一个字的概率 batch_size
        for i in range(1,len(lab[0])):
            if attention_mask[0][i]==1:
                score=score+self.transitions[lab[:,i-1],lab[:,i]]+l_matrix[torch.arange(1,dtype=torch.long),i,lab[:,i]] #batch_size
        # score+=self.transitions[lab[:,i],26]
        # score+=self.transitions[25,lab[:,0]]
        return score
    def score_total(self,l_matrix,attention_mask):
        score=torch.zeros_like(l_matrix[:,0])
        tran=self.transitions
        for i in range(1,len(l_matrix[0])):
            if attention_mask[0][i]==1:
                score=score.unsqueeze(2)+tran+l_matrix[:,i].unsqueeze(1)
                score=torch.logsumexp(score,dim=1)
        # start=torch.logsumexp(self.transitions[25,:],dim=0)
        # end=torch.logsumexp(self.transitions[:,26],dim=0)
        score=torch.logsumexp(score,dim=1)
        return score
    def loss(self,sent,attention_mask,lab):
        l_matrix=self.likely_matrix(sent,attention_mask)
        score_sentence=self.score_sentence(lab,l_matrix, attention_mask)
        score_total=self.score_total(l_matrix,attention_mask)
        return score_total-score_sentence
    def viterbi_decode(self,l_matrix,attention_mask):
        transitions=self.transitions
        l_matrix=l_matrix[0]
        t_score=torch.zeros_like(l_matrix)
        t_score[0]=l_matrix[0]+transitions[25,0:27]
        backpointers=torch.zeros_like(l_matrix,dtype=int)
        for i in range(1,len(l_matrix)):
            if i<len(l_matrix)-1 and attention_mask[0][i+1]==0:
                break
            
            t=t_score[i-1].unsqueeze(1)+transitions
            t_score[i]=max(t,axis=0)+l_matrix[i]
            backpointers[i]=argmax(t,axis=0)
                
        t_score[-1]=t_score[-1]+transitions[:,26]
        path_score=max(t_score[-1],0)
        best_path=[argmax(t_score[-1],0)]
        for i in range(len(t_score)-1,0,-1):
            best_path.append(backpointers[i,best_path[-1]])
        best_path.reverse()
        return path_score, best_path
    def forward(self,sent,attention_mask):
        l_matrix=self.likely_matrix(sent,attention_mask)
        path_score,path_index=self.viterbi_decode(l_matrix,attention_mask)
        return path_score,path_index