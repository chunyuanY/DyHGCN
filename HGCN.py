import torch
import torch.nn as nn
# from ..inits import glorot, zeros
import torch.nn.init as init
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_scatter import scatter_add
import torch.nn.functional as F
import Constants
from DataConstruct import LoadDynamicHeteGraph


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
        graph_x_embeddings = self.gnn1(self.embedding.weight, graph_edge_index)
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)
        return graph_output


class HeteConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, num_relations,
                       cached=False, improved=False, bias=True, **kwargs):
        super(HeteConv, self).__init__(aggr='add', **kwargs)

        self.cached = cached
        self.improved = improved
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None, edge_weight=None):
        '''
        :param x:
        :param edge_index:
        :param edge_type:
        :param edge_norm:
        :param size:
        :return:
        '''
        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=norm)

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        # fill_value = 1 if not improved else 2
        # edge_index, edge_weight = add_remaining_self_loops(
        #     edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def message(self, edge_index, x_j, edge_type, edge_norm):
        '''
        :param x_i: [E, in_channels]
        :param x_j: [E, in_channels]
        :param edge_index_j: [E, ]
        :param edge_type: [E, ]
        :param edge_norm:
        :return:
        '''
        t1 = edge_norm * edge_type.float()
        msg1 = t1.view(-1, 1) * x_j

        t2 = edge_norm * (1 - edge_type.float())
        msg2 = t2.view(-1, 1) * x_j

        msg = 0.1*msg1 + 0.9 * msg2

        # return out if edge_norm is None else out * edge_norm.view(-1, 1)
        return msg

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class DyHGCN(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(DyHGCN, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp, padding_idx=0)
        init.xavier_normal_(self.embedding.weight)
        self.gnn1 = HeteConv(ninp, ninp, num_relations=2)
        self.gnn2 = HeteConv(ninp, ninp, num_relations=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, input_timestamp, step_len=5):
        batch_size, max_len = input.size()
        outputs = torch.zeros(batch_size, max_len, self.nhid).cuda()

        for t in range(0, max_len, step_len):
            la_timestamp = torch.max(input_timestamp[:, t:t+step_len]).item()
            dynamic_graph = LoadDynamicHeteGraph(Constants.data_path, la_timestamp)

            edge_index = dynamic_graph.edge_index.cuda()
            edge_type = dynamic_graph.edge_type.cuda()
            edge_weight = dynamic_graph.edge_weight.cuda()

            graph_x_embeddings = self.gnn1(self.embedding.weight, edge_index, edge_type, edge_weight=edge_weight)
            graph_x_embeddings = self.dropout(graph_x_embeddings)
            graph_dynamic_embeddings = self.gnn2(graph_x_embeddings, edge_index, edge_type, edge_weight=edge_weight)

            outputs[:, t:t+step_len, :] = F.embedding(input[:, t:t+step_len], graph_dynamic_embeddings)
        return outputs


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





