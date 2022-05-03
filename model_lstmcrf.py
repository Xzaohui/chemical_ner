import torch
import torch.nn as nn
import pre_data
class Lstm_model(nn.Module):
    def __init__(self):
        super(Lstm_model,self).__init__()
        self.embedding = nn.Embedding(23623,200)
        self.lstm = nn.LSTM(200,256,2)
        self.linear = nn.Linear(256,5)
        self.transitions=nn.Parameter(torch.empty(7,7))
        nn.init.uniform_(self.transitions, 0, 1) 
        self.transitions.data[:,pre_data.lab['start']]=-1000
        self.transitions.data[pre_data.lab['end'],:]=-1000
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
        score=l_matrix[torch.arange(1,dtype=torch.long),0,lab[:,0]]
        for i in range(1,len(lab[0])):
            score=score+self.transitions[lab[:,i-1],lab[:,i]]+l_matrix[torch.arange(1,dtype=torch.long),i,lab[:,i]]
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
    def viterbi_decode(self,sent):
        1
    
    
