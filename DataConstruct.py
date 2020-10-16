''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
from torch.autograd import Variable
import Constants
import logging
import pickle
import os 
import pandas as pd
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

class Options(object):
    
    def __init__(self, data_name = 'twitter'):
        #data options.
        #data_name = 'twitter'
        #train file path.
        self.train_data = data_name+'/cascade.txt'
        #valid file path.
        self.valid_data = data_name+'/cascadevalid.txt'
        #test file path.
        self.test_data = data_name+'/cascadetest.txt'

        self.u2idx_dict = data_name+'/u2idx.pickle'

        self.idx2u_dict = data_name+'/idx2u.pickle'
        #save path.
        self.save_path = ''

        self.batch_size = 32

        self.net_data = data_name+'/edges.txt'

        self.embed_dim = 64
        # self.embed_file = data_name+'/dw'+str(self.embed_dim)+'.txt'

def LoadRelationGraph(data_name):
    options = Options(data_name)
    _u2idx = {}
    _idx2u = [] 

    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)
    with open(options.idx2u_dict, 'rb') as handle:
        _idx2u = pickle.load(handle)

    edges_list = []
    if os.path.exists(options.net_data):
        with open(options.net_data, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edge.split(',') for edge in relation_list]
            relation_list = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in relation_list if edge[0] in _u2idx and edge[1] in _u2idx]
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            edges_list += relation_list_reverse
    else:
        return [] 
    edges_list_tensor = torch.LongTensor(edges_list).t()
    edges_weight = torch.ones(edges_list_tensor.size(1)).float()

    data = Data(edge_index=edges_list_tensor, edge_attr=edges_weight)
    return data


def LoadDynamicDiffusionGraph(data_name, time_step_split=Constants.time_step_split):
    if not os.path.exists(data_name + "diffusion_graph.csv"):
        options = Options(data_name)
        _u2idx = {}
        _idx2u = []

        with open(options.u2idx_dict, 'rb') as handle:
            _u2idx = pickle.load(handle)
        with open(options.idx2u_dict, 'rb') as handle:
            _idx2u = pickle.load(handle)

        with open(options.train_data, 'r') as handle:
            cascade_list = handle.read().strip().split("\n")
            cascade_list = [chunk.strip().split(" ") for chunk in cascade_list if len(chunk.strip()) != 0]

        t_cascades = []
        for chunks in cascade_list:
            userlist = [chunk.split(',') for chunk in chunks]
            userlist = [[_u2idx[x[0]], int(x[1])] for x in userlist if x[0] in _u2idx]
            pair_user = [(i[0], j[0], j[1]) for i, j in zip(userlist[::1], userlist[1::1])]
            if len(pair_user) > 1 and len(pair_user) <= 500:
                t_cascades += pair_user

        t_cascades_pd = pd.DataFrame(t_cascades)
        t_cascades_pd.columns = ["user1", "user2", "timestamp"]
        t_cascades_pd.to_csv(data_name + "diffusion_graph.csv", index=False)
    else:
        t_cascades_pd = pd.read_csv(data_name + "diffusion_graph.csv")

    t_cascades_pd = t_cascades_pd.sort_values(by="timestamp")
    t_cascades_length = t_cascades_pd.shape[0]
    step_length_x = t_cascades_length // time_step_split

    t_cascades_list = dict() 
    for x in range(step_length_x, t_cascades_length-step_length_x, step_length_x):
        t_cascades_pd_sub = t_cascades_pd[:x]
        t_cascades_sub_list = t_cascades_pd_sub.apply(lambda x: (x["user1"], x["user2"]), axis=1).tolist()
        sub_timesas = t_cascades_pd_sub["timestamp"].max()
        # t_cascades_list.append(t_cascades_sub_list)
        t_cascades_list[sub_timesas] = t_cascades_sub_list
    # + all
    t_cascades_sub_list = t_cascades_pd.apply(lambda x: (x["user1"], x["user2"]), axis=1).tolist()
    # t_cascades_list.append(t_cascades_sub_list) 
    sub_timesas = t_cascades_pd["timestamp"].max()
    t_cascades_list[sub_timesas] = t_cascades_sub_list

    dynamic_graph_dict_list = dict() 
    for key in sorted(t_cascades_list.keys()): 
        edges_list = t_cascades_list[key]
        # edges_list_tensor = torch.LongTensor(edges_list).t() 
        # loader = DataLoader(dataset, batch_size=32, shuffle=True) 
        # data = Data(edge_index=edges_list_tensor)
        cascade_dic = defaultdict(list)
        for upair in edges_list:
            cascade_dic[upair].append(1) 
        dynamic_graph_dict_list[key] = cascade_dic
    return dynamic_graph_dict_list 
    #######################
    t_cascades_list = list()
    for x in range(step_length_x, t_cascades_length - step_length_x, step_length_x):
        t_cascades_pd_sub = t_cascades_pd[:x]
        t_cascades_sub_list = t_cascades_pd_sub.apply(lambda x: (x["user1"], x["user2"]), axis=1).tolist()
        t_cascades_list.append(t_cascades_sub_list)

    # + all
    t_cascades_sub_list = t_cascades_pd.apply(lambda x: (x["user1"], x["user2"]), axis=1).tolist()
    t_cascades_list.append(t_cascades_sub_list) 

    diffusion_dic_list = []
    for repost_relation in t_cascades_list:
        cascade_dic = defaultdict(list)
        for upair in repost_relation:
            cascade_dic[upair].append(1)
        diffusion_dic_list.append(cascade_dic)

    return diffusion_dic_list


def LoadDynamicHeteGraph(data_name):
    options = Options(data_name)
    _u2idx = {}
    _idx2u = []

    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)
    with open(options.idx2u_dict, 'rb') as handle:
        _idx2u = pickle.load(handle)

    follow_relation = []  # directed relation
    if os.path.exists(options.net_data):
        with open(options.net_data, 'r') as handle:
            edges_list = handle.read().strip().split("\n")
            edges_list = [edge.split(',') for edge in edges_list]
            follow_relation = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in edges_list if edge[0] in _u2idx and edge[1] in _u2idx]

    dy_diff_graph_list = LoadDynamicDiffusionGraph(data_name)
    dynamic_graph = dict()
    for x in sorted(dy_diff_graph_list.keys()):
        edges_list = follow_relation
        edges_type_list = [0] * len(follow_relation)  # 0:follow relation,  1:repost relation
        edges_weight = [1.0] * len(follow_relation)
        for key, value in dy_diff_graph_list[x].items():
            edges_list.append(key)
            edges_type_list.append(1)
            edges_weight.append(sum(value))

        edges_list_tensor = torch.LongTensor(edges_list).t()
        edges_type = torch.LongTensor(edges_type_list)
        edges_weight = torch.FloatTensor(edges_weight)

        data = Data(edge_index=edges_list_tensor, edge_type=edges_type, edge_weight=edges_weight)
        dynamic_graph[x] = data 
    return dynamic_graph


def LoadDynamicHeteGraphWithoutSocialGraph(data_name):
    options = Options(data_name)
    _u2idx = {}
    _idx2u = []

    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)
    with open(options.idx2u_dict, 'rb') as handle:
        _idx2u = pickle.load(handle)

    dy_diff_graph_list = LoadDynamicDiffusionGraph(data_name)
    dynamic_graph = dict()
    for x in sorted(dy_diff_graph_list.keys()):
        edges_list = [] 
        edges_type_list = [] 
        edges_weight = [] 
        for key, value in dy_diff_graph_list[x].items():
            edges_list.append(key)
            edges_type_list.append(1)
            edges_weight.append(sum(value))

        edges_list_tensor = torch.LongTensor(edges_list).t()
        edges_type = torch.LongTensor(edges_type_list)
        edges_weight = torch.FloatTensor(edges_weight)

        data = Data(edge_index=edges_list_tensor, edge_type=edges_type, edge_weight=edges_weight)
        dynamic_graph[x] = data 
    return dynamic_graph

class DataConstruct(object):
    ''' For data iteration '''

    def __init__(
            self, data_name, data=0, load_dict=True, cuda=True, batch_size=32, shuffle=True, test=False, with_EOS=True): #data = 0 for train, 1 for valid, 2 for test
        self.options = Options(data_name)
        self.options.batch_size = batch_size
        self._u2idx = {}
        self._idx2u = []
        self.data = data
        self.test = test
        self.with_EOS = with_EOS
        if not load_dict:
            self._buildIndex()
            with open(self.options.u2idx_dict, 'wb') as handle:
                pickle.dump(self._u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.options.idx2u_dict, 'wb') as handle:
                pickle.dump(self._idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.options.u2idx_dict, 'rb') as handle:
                self._u2idx = pickle.load(handle)
            with open(self.options.idx2u_dict, 'rb') as handle:
                self._idx2u = pickle.load(handle)
            self.user_size = len(self._u2idx)
        self._train_cascades,train_len = self._readFromFile(self.options.train_data)
        self._valid_cascades,valid_len = self._readFromFile(self.options.valid_data)
        self._test_cascades,test_len = self._readFromFile(self.options.test_data)
        self._train_cascades_timestamp = self._readFromFileTimestamp(self.options.train_data)
        self._valid_cascades_timestamp = self._readFromFileTimestamp(self.options.valid_data)
        self._test_cascades_timestamp = self._readFromFileTimestamp(self.options.test_data)

        self.train_size = len(self._train_cascades)
        self.valid_size = len(self._valid_cascades)
        self.test_size = len(self._test_cascades)
        self.cuda = cuda

        if self.data == 0:
            self._n_batch = int(np.ceil(len(self._train_cascades) / batch_size))
        elif self.data == 1:
            self._n_batch = int(np.ceil(len(self._valid_cascades) / batch_size))
        else:
            self._n_batch = int(np.ceil(len(self._test_cascades) / batch_size))

        self._batch_size = self.options.batch_size

        self._iter_count = 0

        self._need_shuffle = shuffle

        if self._need_shuffle:
            random_seed_int = random.randint(0, 1000)
            random.seed(random_seed_int)
            random.shuffle(self._train_cascades)
            random.seed(random_seed_int)
            random.shuffle(self._train_cascades_timestamp)

    def _buildIndex(self):
        #compute an index of the users that appear at least once in the training and testing cascades.
        opts = self.options

        train_user_set = set()
        valid_user_set = set()
        test_user_set = set()

        lineid=0
        for line in open(opts.train_data):
            lineid+=1
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                train_user_set.add(user)

        for line in open(opts.valid_data):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                valid_user_set.add(user)

        for line in open(opts.test_data):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                test_user_set.add(user)

        user_set = train_user_set | valid_user_set | test_user_set

        pos = 0
        self._u2idx['<blank>'] = pos
        self._idx2u.append('<blank>')
        pos += 1
        self._u2idx['</s>'] = pos
        self._idx2u.append('</s>')
        pos += 1

        for user in user_set:
            self._u2idx[user] = pos
            self._idx2u.append(user)
            pos += 1
        opts.user_size = len(user_set) + 2
        self.user_size = len(user_set) + 2
        print("user_size : %d" % (opts.user_size))

    def _readNet(self, filename):
        adj_list=[[],[],[]]
        n_edges = 0
        # add self edges
        for i in range(self.user_size):
            adj_list[0].append(i)
            adj_list[1].append(i)
            adj_list[2].append(1)
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            nodes = line.strip().split(',')
            if nodes[0] not in self._u2idx.keys() or nodes[1] not in self._u2idx.keys():
                continue
            n_edges+=1
            adj_list[0].append(self._u2idx[nodes[0]])
            adj_list[1].append(self._u2idx[nodes[1]])
            adj_list[2].append(1) # weight
        # print('edge:', n_edges/2)
        return adj_list

    def _readNet_dict_list(self, filename):
        adj_list={}
        # add self edges
        for i in range(self.user_size):
            adj_list.setdefault(i,[i]) # [i] or []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            nodes = line.strip().split(',')
            if nodes[0] not in self._u2idx.keys() or nodes[1] not in self._u2idx.keys():
                continue
            adj_list[self._u2idx[nodes[0]]].append(self._u2idx[nodes[1]])
            adj_list[self._u2idx[nodes[1]]].append(self._u2idx[nodes[0]])
        return adj_list

    def _load_ne(self, filename, dim):
        embed_file=open(filename,'r')
        line = embed_file.readline().strip()
        dim = int(line.split()[1])
        embeds = np.zeros((self.user_size,dim))
        for line in embed_file.readlines():
            line=line.strip().split()
            embeds[self._u2idx[line[0]],:]= np.array(line[1:])
        return embeds

    def _readFromFile(self, filename):
        """read all cascade from training or testing files. """
        total_len = 0
        t_cascades = []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            userlist = []
            chunks = line.strip().split()
            for chunk in chunks:
                #try:
                user, timestamp = chunk.split(',')
                # except:
                #     print(chunk)
                if user in self._u2idx:
                    userlist.append(self._u2idx[user])

            if len(userlist) > 1 and len(userlist)<=500:
                total_len+=len(userlist)
                if self.with_EOS:
                    userlist.append(Constants.EOS)
                t_cascades.append(userlist)
        return t_cascades,total_len

    def _readFromFileTimestamp(self, filename):
        """read all cascade from training or testing files. """
        t_cascades = []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            timestamplist = []
            chunks = line.strip().split()
            for chunk in chunks:
                # try:
                user, timestamp = chunk.split(',')
                timestamp = int(timestamp)
                # timestamp = timestamp // (60 * 60 * 24)
                # except:
                #     print(chunk)
                if user in self._u2idx:
                    timestamplist.append(timestamp)

            if len(timestamplist) > 1 and len(timestamplist)<=500:
                if self.with_EOS:
                    timestamplist.append(Constants.EOS)
                t_cascades.append(timestamplist)
        return t_cascades

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''

            max_len = max(len(inst) for inst in insts)

            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst))
                for inst in insts])
        
            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()

            return inst_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            if self.data == 0:
                seq_insts = self._train_cascades[start_idx:end_idx]
                seq_timestamp = self._train_cascades_timestamp[start_idx:end_idx]
            elif self.data == 1:
                seq_insts = self._valid_cascades[start_idx:end_idx]
                seq_timestamp = self._valid_cascades[start_idx:end_idx]
            else:
                seq_insts = self._test_cascades[start_idx:end_idx]
                seq_timestamp = self._test_cascades_timestamp[start_idx:end_idx]
            seq_data = pad_to_longest(seq_insts)
            seq_data_timestamp = pad_to_longest(seq_timestamp)
            
            return seq_data, seq_data_timestamp
        else:

            if self._need_shuffle:
                random.shuffle(self._train_cascades)
                #random.shuffle(self._test_cascades)

            self._iter_count = 0
            raise StopIteration()
