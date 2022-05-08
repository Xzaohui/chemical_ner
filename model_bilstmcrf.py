import torch
import torch.nn as nn
import pre_data
def argmax(vec,axis):
    # return the argmax as a python int
    _, idx = torch.max(vec, axis)
    return idx
def max(vec,axis):
    # return the argmax as a python int
    max, _ = torch.max(vec, axis)
    return max
class Lstm_model(nn.Module):
    def __init__(self):
        super(Lstm_model,self).__init__()
        self.embedding = nn.Embedding(len(pre_data.dic),200)
        self.lstm = nn.LSTM(200,32,2,bidirectional=True)
        self.linear = nn.Linear(64,5)
        self.transitions=nn.Parameter(torch.empty(7,7))
        nn.init.uniform_(self.transitions, 0, 1) 
        self.transitions.data[:,pre_data.lab['start']]=-1000
        self.transitions.data[pre_data.lab['end'],:]=-1000
        self.transitions.data[pre_data.lab['start'],pre_data.lab['end']]=-1000
    def likely_matrix(self,sent):
        sent=self.embedding(sent)
        sent,_ = self.lstm(sent)
        score = self.linear(sent)
        return score
    def loss(self,sent,lab):
        l_matrix=self.likely_matrix(sent)
        score_sentence=self.score_sentence(lab,l_matrix)
        score_total=self.score_total(l_matrix)
        return score_total-score_sentence
    def score_sentence(self,lab,l_matrix):
        score=l_matrix[torch.arange(1,dtype=torch.long),0,lab[:,0]] #第一个字的概率 batch_size
        for i in range(1,len(lab[0])):
            score=score+self.transitions[lab[:,i-1],lab[:,i]]+l_matrix[torch.arange(1,dtype=torch.long),i,lab[:,i]] #batch_size
        score+=self.transitions[lab[:,i],pre_data.lab['end']]
        score+=self.transitions[pre_data.lab['start'],lab[:,0]]
        return score
    def score_total(self,l_matrix):
        score=l_matrix[:,0]
        tran=self.transitions[0:5,0:5]
        for i in range(1,len(l_matrix[0])):
            score=score.unsqueeze(2)+tran+l_matrix[:,i].unsqueeze(1)
            score=torch.logsumexp(score,dim=1)
        start=torch.logsumexp(self.transitions[pre_data.lab['start'],:],dim=0)
        end=torch.logsumexp(self.transitions[:,pre_data.lab['end']],dim=0)
        score=torch.logsumexp(score,dim=1)+start+end
        return score
    def viterbi_decode(self,l_matrix):
        transitions=self.transitions
        tran=transitions[0:5,0:5]
        l_matrix=l_matrix[0]
        t_score=torch.zeros_like(l_matrix)
        t_score[0]=l_matrix[0]+transitions[pre_data.lab['start'],0:5]
        backpointers=torch.zeros_like(l_matrix,dtype=int)
        for i in range(1,len(l_matrix)):
            t=t_score[i-1].unsqueeze(1)+tran
            t_score[i]=max(t,axis=0)+l_matrix[i]
            backpointers[i]=argmax(t,axis=0)
        t_score[-1]=t_score[-1]+transitions[0:5,pre_data.lab['end']]
        path_score=max(t_score[-1],0)
        best_path=[argmax(t_score[-1],0)]
        for i in range(len(t_score)-1,0,-1):
            best_path.append(backpointers[i,best_path[-1]])
        best_path.reverse()
        return path_score, best_path
    def forward(self,sent):
        l_matrix=self.likely_matrix(sent)
        path_score,path_index=self.viterbi_decode(l_matrix)
        return path_score,path_index
    
    
