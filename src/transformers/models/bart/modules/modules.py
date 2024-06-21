import torch
import torch.nn.functional as F
from torch import nn
import torch
import torch.nn.functional as F
from torch import nn

class EmoTrans(nn.Module):
    def __init__(self, n_emo_in, n_emo_out, n_strat, embed_dim):
        super().__init__()
        self.n_emo_in = n_emo_in
        self.n_emo_out = n_emo_out
        self.n_strat = n_strat
        self.embed_dim = embed_dim
        self.matrices = nn.ParameterList([nn.Parameter(torch.Tensor(n_emo_in, n_emo_out)) for i in range(self.n_strat )])
        self.emotion_embedding = nn.Embedding(n_emo_out, embed_dim)
        self.emotion_id = torch.tensor(range(n_emo_out), dtype=torch.long)
        self.dropout = nn.Dropout(0.1)
        self.reset_weights()
    def reset_weights(self):
        for weight in self.matrices:
            torch.nn.init.ones_(
                weight)
    def forward(self, emo_logits, strat_logits, stop_norm_weight = False):
        b = emo_logits.size(0)
        emo_out_logits_each_strat = torch.zeros(b, self.n_strat, self.n_emo_out).to(emo_logits.device) #[b, stra, emo]
        emo_logits = self.dropout(emo_logits)
        strat_logits = self.dropout(strat_logits)
        emo_prob = F.softmax(emo_logits, dim = -1)
        for i,matrix in enumerate(self.matrices):
            if stop_norm_weight:
                emo_out_logits_cur_strat = F.softmax(F.linear(emo_prob, matrix.t()))
            else:
                with torch.no_grad():
                    weight_norm = matrix/matrix.sum(dim=1, keepdim=True)
                    matrix.copy_(weight_norm)
                emo_out_logits_cur_strat = F.linear(emo_prob, matrix.t())
            emo_out_logits_each_strat[:, i, :] = emo_out_logits_cur_strat
        #for i in range(len(self.matrices)):
        #    with torch.no_grad():
        #        weight_norm = self.matrices[i]/self.matrices[i].sum(dim=1, keepdim=True)
        #        self.matrices[i].copy_(weight_norm)
        #    emo_out_logits_cur_strat = F.linear(emo_prob, self.matrices[i].t())
        #    emo_out_logits_each_strat[:, i, :] = emo_out_logits_cur_strat
        strat_prob = F.softmax(strat_logits, dim = -1)
        emo_out_prob = torch.bmm(strat_prob.unsqueeze(-2), emo_out_logits_each_strat) #[b, 1, stra] * [b, stra, emo] -> [b, 1, emo] 
        emotion_id = self.emotion_id.to(emo_logits.device) 
        emo_embed = torch.bmm(emo_out_prob,  self.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1))
        emo_out_prob = emo_out_prob.squeeze()
        emo_out_prob = torch.log(emo_out_prob) #upDATE  9-27-II
        return emo_embed, emo_out_prob
        
        

class EmoTrans_wo_STRA(nn.Module):
    def __init__(self, n_emo_in, n_emo_out, n_strat, embed_dim):
        super().__init__()
        self.n_emo_in = n_emo_in
        self.n_emo_out = n_emo_out
        self.n_strat = 1
        self.embed_dim = embed_dim
        self.matrices = nn.ParameterList([nn.Parameter(torch.Tensor(n_emo_in, n_emo_out)) for i in range(self.n_strat )])
        self.emotion_embedding = nn.Embedding(n_emo_out, embed_dim)
        self.emotion_id = torch.tensor(range(n_emo_out), dtype=torch.long)
        self.dropout = nn.Dropout(0.1)
        self.reset_weights()
    def reset_weights(self):
        for weight in self.matrices:
            torch.nn.init.ones_(
                weight)
    def forward(self, emo_logits, strat_logits):
        b = emo_logits.size(0)
        emo_out_logits_each_strat = torch.zeros(b, self.n_strat, self.n_emo_out).to(emo_logits.device) #[b, stra, emo]
        emo_logits = self.dropout(emo_logits)
        strat_logits = self.dropout(strat_logits)
        emo_prob = F.softmax(emo_logits, dim = -1)
        for i,matrix in enumerate(self.matrices):
            with torch.no_grad():
                weight_norm = matrix/matrix.sum(dim=1, keepdim=True)
                matrix.copy_(weight_norm)
            emo_out_logits_cur_strat = F.linear(emo_prob, matrix.t())
            emo_out_logits_each_strat[:, i, :] = emo_out_logits_cur_strat
        #strat_prob = F.softmax(strat_logits, dim = -1)
        emo_out_prob =  emo_out_logits_each_strat #[b, 1, emo_out]
        emotion_id = self.emotion_id.to(emo_logits.device) 
        emo_embed = torch.bmm(emo_out_prob,  self.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1))
        emo_out_prob = emo_out_prob.squeeze()
        emo_out_prob = torch.log(emo_out_prob) #upDATE  9-27-II
        return emo_embed, emo_out_prob

class EmoTrans_wo_Emo(nn.Module):
    def __init__(self, n_emo_in, n_emo_out, n_strat, embed_dim):
        super().__init__()
        self.n_emo_in = n_emo_in
        self.n_emo_out = n_emo_out
        self.n_strat = n_strat
        self.embed_dim = embed_dim
        self.matrix = nn.Parameter(torch.Tensor(n_strat, n_emo_out))
        self.emotion_embedding = nn.Embedding(n_emo_out, embed_dim)
        self.emotion_id = torch.tensor(range(n_emo_out), dtype=torch.long)
        self.dropout = nn.Dropout(0.1)
        self.reset_weights()
    def reset_weights(self):

        torch.nn.init.ones_(
                self.matrix)
    def forward(self, emo_logits, strat_logits):
        b = emo_logits.size(0)
        #emo_out_logits_each_strat = torch.zeros(b, self.n_strat, self.n_emo_out).to(emo_logits.device) #[b, stra, emo]
        #emo_logits = self.dropout(emo_logits)
        strat_logits = self.dropout(strat_logits)
        strat_prob = F.softmax(strat_logits, dim = -1)
        if len(strat_prob.size()) == 2:
            strat_prob = strat_prob.unsqueeze(-2)
        #emo_prob = F.softmax(emo_logits, dim = -1)
        with torch.no_grad():
            weight_norm = self.matrix/self.matrix.sum(dim=1, keepdim=True)
            self.matrix.copy_(weight_norm)
        emo_out_logits = F.linear(strat_prob, self.matrix.t())
            
        #strat_prob = F.softmax(strat_logits, dim = -1)
        emo_out_prob =  F.softmax(emo_out_logits, dim = -1) #[b, 1, emo_out]
        emotion_id = self.emotion_id.to(emo_logits.device) 
        emo_embed = torch.bmm(emo_out_prob,  self.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1))
        emo_out_prob = emo_out_prob.squeeze()
        emo_out_prob = torch.log(emo_out_prob) #upDATE  9-27-II
        return emo_embed, emo_out_prob



class ContrastiveLoss(nn.Module):
    def __init__(self, m=2.0):
        super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
        self.m = m  # margin or radius

    def pair_forward(self, y1, y2, d=0):
        # d = 1 means y1 and y2 are supposed to be same
        # d = 0 means y1 and y2 are supposed to be different
        
        #euc_dist = nn.functional.pairwise_distance(y1, y2)
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)(y1, y2)
        #cos = torch.tensor(cos).to(y1.device)
        #print(cos)
        #print(cos.shape)
        if d == 0:#not same 
            return torch.mean(torch.pow(cos, 2))  # distance squared
        else:  # d == 1 #same 
            delta = self.m - cos  # sort of reverse distance
            delta = torch.clamp(delta, min=0.0, max=None)
            return torch.mean(torch.pow(delta, 2))  # mean over all rows
    def forward(self, ys, labels):
        b = ys.size(0)
        contrast_loss = 0
        n_pair = 0
        #n_pair = ((b-1) * b)/2
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i > j:
                    y_1 = ys[i]
                    y_2 = ys[j]
                    y_1_label = labels[i]
                    y_2_label = labels[j]
                    with torch.no_grad():
                        d = (y_1_label == y_2_label).float().item()
                    loss = self.pair_forward(y_1, y_2, d)
                    contrast_loss += loss
                    n_pair += 1
        
        return contrast_loss / n_pair
                
        
def get_last_arg_where_equal(x, value):
    ts = torch.tensor(x)
    nzr = (ts == value).nonzero()
    i_pos = (nzr[:,0].diff() == 1).nonzero().squeeze(-1)
    i_pos = torch.cat((i_pos, torch.tensor([nzr.size(0) - 1]).to(i_pos.device)))
    pos = nzr[i_pos,:]
    return pos

if __name__ == "__main__":
    n_emo_in = 3
    n_emo_out = 4
    n_strat = 5
    batch_size = 2
    tran = EmoTrans(n_emo_in, n_emo_out, n_strat, embed_dim = 32)
    emo_logits = torch.full((batch_size, n_emo_in), 3.1)
    strat_logits = torch.full((batch_size, n_strat), 2.2)
    emo_embed, emo_out_logits = tran(emo_logits, strat_logits)