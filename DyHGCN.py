
import numpy as np
import time 
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


import Constants
from TransformerBlock import TransformerBlock

"""
    Soft Attention 
"""


class GraphNN(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(GraphNN, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp, padding_idx=0)

        self.gnn1 = GCNConv(ninp, ninp * 2)
        self.gnn2 = GCNConv(ninp * 2, ninp)

        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)
        init.xavier_normal_(self.gnn1.weight)
        init.xavier_normal_(self.gnn2.weight)

    def forward(self, graph):
        graph_edge_index = graph.edge_index.cuda()
        # print (graph_x_embeddings.shape)
        graph_x_embeddings = self.gnn1(self.embedding.weight, graph_edge_index)
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)
        return graph_output.cuda()


class DynamicGraphNN(nn.Module):
    def __init__(self, ntoken, nhid, dropout=0.1):
        super(DynamicGraphNN, self).__init__()
        self.nhid = nhid
        self.ntoken = ntoken
        self.embedding = nn.Embedding(ntoken, nhid)
        init.xavier_normal_(self.embedding.weight)

        self.gnn1 = GraphNN(ntoken, nhid)
        self.linear = nn.Linear(nhid * Constants.time_step_split, nhid) 
        init.xavier_normal_(self.linear.weight)
        self.drop = nn.Dropout(dropout)

    def forward(self, diffusion_graph_list):
        res = dict()
        graph_embeddinng_list = list() 
        for key in sorted(diffusion_graph_list.keys()):
            graph = diffusion_graph_list[key] 
            graph_x_embeddings = self.gnn1(graph)
            graph_x_embeddings = self.drop(graph_x_embeddings)
            graph_x_embeddings = graph_x_embeddings.cpu()

            graph_embeddinng_list.append(graph_x_embeddings)
            res[key] = graph_x_embeddings
        return res 

class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)

def get_previous_user_mask(seq, user_size):
    ''' Mask previous activated users.'''
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()

    masked_seq = previous_mask * seqs.data.float()
    # print(masked_seq.size())

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
    masked_seq = Variable(masked_seq, requires_grad=False)
    return masked_seq


class TimeAttention(nn.Module):
    def __init__(self, time_size, in_features1):
        super(TimeAttention, self).__init__()
        self.time_embedding = nn.Embedding(time_size, in_features1)
        init.xavier_normal_(self.time_embedding.weight)
        self.dropout = nn.Dropout(0.1) 

    def forward(self, T_idx, Dy_U_embed, mask=None, episilon=1e-6):
        '''
            T_idx: (bsz, user_len)
            Dy_U_embed: (bsz, user_len, time_len, d) # uid 从动态embedding lookup 之后的节点向量
            output: (bsz, user_len, d) 
        '''
        temperature = Dy_U_embed.size(-1) ** 0.5 + episilon
        T_embed = self.time_embedding(T_idx) # (bsz, user_len, d)

        # print(T_embed.size())
        # print(Dy_U_embed.size())

        affine = torch.einsum("bud,butd->but", T_embed, Dy_U_embed) # (bsz, user_len, time_len)
        score = affine / temperature 

        # if mask is None:
        #     mask = torch.triu(torch.ones(score.size()), diagonal=1).bool().cuda()
        #     score = score.masked_fill(mask, -2**32+1)

        alpha = F.softmax(score, dim=1)  # (bsz, user_len, time_len) 
        # alpha = self.dropout(alpha)
        alpha = alpha.unsqueeze(dim=-1)  # (bsz, user_len, time_len, 1) 

        att = (alpha * Dy_U_embed).sum(dim=2)  # (bsz, user_len, d) 
        return att 



class DyHGCN_S(nn.Module):

    def __init__(self, opt, dropout=0.1):
        super(DyHGCN_S, self).__init__()
        ntoken = opt.user_size
        ninp = opt.d_word_vec
        self.ninp = ninp
        self.user_size = ntoken
        self.pos_dim = 8
        self.__name__ = "DyHGCN_S"

        self.dropout = nn.Dropout(dropout)
        self.drop_timestamp = nn.Dropout(dropout)

        self.gnn_layer = GraphNN(ntoken, ninp)
        self.gnn_diffusion_layer = DynamicGraphNN(ntoken, ninp)
        self.pos_embedding = nn.Embedding(1000, self.pos_dim)

        self.time_attention = TimeAttention(Constants.time_step_split, self.ninp)
        self.decoder_attention = TransformerBlock(input_size=ninp + self.pos_dim, n_heads=8)
        self.linear = nn.Linear(ninp + self.pos_dim, ntoken)
        self.init_weights()
        print(self)

    def init_weights(self):
        init.xavier_normal_(self.pos_embedding.weight)
        init.xavier_normal_(self.linear.weight)

    def forward(self, input, input_timestamp, relation_graph, diffusion_graph):
        input = input[:, :-1]
        mask = (input == Constants.PAD)

        batch_t = torch.arange(input.size(1)).expand(input.size()).cuda()
        order_embed = self.dropout(self.pos_embedding(batch_t))

        batch_size, max_len = input.size()
        dyemb = torch.zeros(batch_size, max_len, self.ninp).cuda()
        input_timestamp = input_timestamp[:, :-1] 
        step_len = 5 
        
        dynamic_node_emb_dict = self.gnn_diffusion_layer(diffusion_graph) #input, input_timestamp, diffusion_graph) 
        dyemb_timestamp = torch.zeros(batch_size, max_len).long()

        dynamic_node_emb_dict_time = sorted(dynamic_node_emb_dict.keys())
        dynamic_node_emb_dict_time_dict = dict()
        for i, val in enumerate(dynamic_node_emb_dict_time):
            dynamic_node_emb_dict_time_dict[val] = i 
        latest_timestamp = dynamic_node_emb_dict_time[-1]
        for t in range(0, max_len, step_len):
            try:
                la_timestamp = torch.max(input_timestamp[:, t:t+step_len]).item()
                if la_timestamp < 1:
                    break 
                latest_timestamp = la_timestamp 
            except Exception:
                pass 

            res_index = len(dynamic_node_emb_dict_time_dict)-1
            for i, val in enumerate(dynamic_node_emb_dict_time_dict.keys()):
                if val <= latest_timestamp:
                    res_index = i  
                    continue 
                else:
                    break
            dyemb_timestamp[:, t:t+step_len] = res_index

        dyuser_emb_list = list() 
        for val in sorted(dynamic_node_emb_dict.keys()):
            dyuser_emb_sub = F.embedding(input.cuda(), dynamic_node_emb_dict[val].cuda()).unsqueeze(2)
            dyuser_emb_list.append(dyuser_emb_sub)
        dyuser_emb = torch.cat(dyuser_emb_list, dim=2)

        dyemb = self.time_attention(dyemb_timestamp.cuda(), dyuser_emb.cuda())
        dyemb = self.dropout(dyemb) 

        final_embed = torch.cat([dyemb, order_embed], dim=-1).cuda() # dynamic_node_emb
        att_out = self.decoder_attention(final_embed.cuda(), final_embed.cuda(), final_embed.cuda(), mask=mask.cuda())
        att_out = self.dropout(att_out.cuda())

        output = self.linear(att_out.cuda())  # (bsz, user_len, |U|)
        mask = get_previous_user_mask(input.cuda(), self.user_size)
        output = output.cuda() + mask.cuda() 

        return output.view(-1, output.size(-1))




class DyHGCN_H(nn.Module):

    def __init__(self, opt, dropout=0.1):
        super(DyHGCN_H, self).__init__()
        ntoken = opt.user_size
        ninp = opt.d_word_vec
        self.ninp = ninp
        self.user_size = ntoken
        self.pos_dim = 8
        self.__name__ = "DyHGCN_H"

        self.dropout = nn.Dropout(dropout)
        self.drop_timestamp = nn.Dropout(dropout)

        self.gnn_layer = GraphNN(ntoken, ninp)
        self.gnn_diffusion_layer = DynamicGraphNN(ntoken, ninp)
        self.pos_embedding = nn.Embedding(1000, self.pos_dim)

        self.time_attention = TimeAttention(Constants.time_step_split, self.ninp)
        self.decoder_attention = TransformerBlock(input_size=ninp + self.pos_dim, n_heads=8)
        self.linear = nn.Linear(ninp + self.pos_dim, ntoken)
        self.init_weights()
        print(self)

    def init_weights(self):
        init.xavier_normal_(self.pos_embedding.weight)
        init.xavier_normal_(self.linear.weight)

    def forward(self, input, input_timestamp, relation_graph, diffusion_graph):
        input = input[:, :-1]
        mask = (input == Constants.PAD)

        batch_t = torch.arange(input.size(1)).expand(input.size()).cuda()
        order_embed = self.dropout(self.pos_embedding(batch_t))

        batch_size, max_len = input.size()
        dyemb = torch.zeros(batch_size, max_len, self.ninp).cuda()
        input_timestamp = input_timestamp[:, :-1] 
        step_len = 5 
        
        dynamic_node_emb_dict = self.gnn_diffusion_layer(diffusion_graph) #input, input_timestamp, diffusion_graph) 
        
        latest_timestamp = sorted(dynamic_node_emb_dict.keys())[-1]
        for t in range(0, max_len, step_len):
            try:
                la_timestamp = torch.max(input_timestamp[:, t:t+step_len]).item()
                if la_timestamp < 1:
                    break 
                latest_timestamp = la_timestamp 
            except Exception:
                # print (input_timestamp[:, t:t+step_len])
                pass 

            his_timestamp = sorted(dynamic_node_emb_dict.keys())[-1]
            for x in sorted(dynamic_node_emb_dict.keys()):
                if x <= latest_timestamp:
                    his_timestamp = x
                    continue
                else:
                    break 

            graph_dynamic_embeddings = dynamic_node_emb_dict[his_timestamp]
            dyemb[:, t:t+step_len, :] = F.embedding(input[:, t:t+step_len].cuda(), graph_dynamic_embeddings.cuda())

        dyemb = self.dropout(dyemb) 

        final_embed = torch.cat([dyemb, order_embed], dim=-1).cuda() # dynamic_node_emb
        att_out = self.decoder_attention(final_embed.cuda(), final_embed.cuda(), final_embed.cuda(), mask=mask.cuda())
        att_out = self.dropout(att_out.cuda())

        output = self.linear(att_out.cuda())  # (bsz, user_len, |U|)
        mask = get_previous_user_mask(input.cuda(), self.user_size)
        output = output.cuda() + mask.cuda() 

        return output.view(-1, output.size(-1))


