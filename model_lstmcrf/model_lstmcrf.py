import torch
import torch.nn as nn
import data_manage

def argmax(vec,axis):
    # return the argmax as a python int
    _, idx = torch.max(vec, axis)
    return idx
def max(vec,axis):
    # return the argmax as a python int
    max, _ = torch.max(vec, axis)
    return max

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec,1).item()]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def len_sent(attention_mask):
    return torch.sum(attention_mask).item()

class Lstm_model(nn.Module):
    def __init__(self):
        super(Lstm_model,self).__init__()
        self.embedding = nn.Embedding(len(data_manage.w2v.wv.index2word),100)
        self.embedding.weight.data.copy_(torch.from_numpy(data_manage.w2v.wv.vectors))
        self.embedding.weight.requires_grad=True
        self.lstm = nn.LSTM(100,256,2)
        self.linear = nn.Linear(256,27)
        self.transitions=nn.Parameter(torch.empty(27,27))
        nn.init.uniform_(self.transitions, 0, 1) 
        self.transitions.data[:,25]=-10000
        self.transitions.data[26,:]=-10000
        self.transitions.data[25,26]=-10000
    def likely_matrix(self,sent):
        sent=self.embedding(sent)
        sent,_ = self.lstm(sent)
        score = self.linear(sent)
        return score
    def score_sentence(self,t_lab,l_matrix,attention_mask):
        l_matrix=l_matrix[0,0:len_sent(attention_mask)]
        lab=[25]+t_lab[0,0:len_sent(attention_mask)].to('cpu').tolist()
        score = torch.zeros(1).to('cuda')
        for i in range(len(l_matrix)):
            score=score+self.transitions[lab[i],lab[i+1]]+l_matrix[i,lab[i+1]] #batch_size
        score+=self.transitions[lab[-1],26]
        return score
    def score_total(self,l_matrix,attention_mask):
        init_alphas = torch.full((1, 27), -10000.).to("cuda")
        init_alphas[0][25] = 0.
        l_matrix=l_matrix[0,0:len_sent(attention_mask)]
        forward_var = init_alphas
        for i in range(len(l_matrix)):
            alphas_t = []
            for next_tag in range(27):
                emit_score = l_matrix[i][next_tag].view(1, -1).expand(1, 27)
                trans_score = self.transitions[:,next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[:,26]
        alpha = log_sum_exp(terminal_var)
        return alpha
    def loss(self,sent,lab,attention_mask):
        l_matrix=self.likely_matrix(sent)
        score_sentence=self.score_sentence(lab,l_matrix, attention_mask)
        score_total=self.score_total(l_matrix,attention_mask)
        return score_total-score_sentence
    def viterbi_decode(self,l_matrix,attention_mask):
        l_matrix=l_matrix[0,0:len_sent(attention_mask)]
        backpointers = []
        init_vvars = torch.full((1, 27), -10000.).to("cuda")
        init_vvars[0][25] = 0
        forward_var = init_vvars
        for feat in l_matrix:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(27):
                next_tag_var = forward_var + self.transitions[:,next_tag]
                best_tag_id = argmax(next_tag_var,1)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        terminal_var = forward_var + self.transitions[:,26]
        best_tag_id = argmax(terminal_var,1)
        path_score = terminal_var[0][best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        assert best_path.pop()==25
        best_path.reverse()
        return path_score, best_path
    def forward(self,sent,attention_mask):
        l_matrix=self.likely_matrix(sent)
        path_score,path_index=self.viterbi_decode(l_matrix,attention_mask)
        return path_score,path_index
    
    
