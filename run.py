import argparse
import time

import numpy as np 
import Constants
import torch
import torch.nn as nn
from DataConstruct import DataConstruct, LoadRelationGraph, LoadDynamicHeteGraph
from Metrics import Metrics
from DyHGCN import DyHGCN_S, DyHGCN_H


from Optim import ScheduledOptim

root_path = './' 

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.cuda.manual_seed(0)

metric = Metrics()


def get_performance(crit, pred, gold):
    ''' Apply label smoothing if needed '''
    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)
    # print ("get performance, ", gold.shape, pred.shape)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
    return loss, n_correct



def train_epoch(model, training_data, graph, diffusion_graph, loss_func, optimizer):
    ''' Epoch operation in training phase'''
    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0
    batch_num = 0.0

    for i, batch in enumerate(training_data): # tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        # prepare data
        tgt, tgt_timestamp = (item.cuda() for item in batch)

        start_time = time.time() 
        import numpy as np
        np.set_printoptions(threshold=np.inf)
        gold = tgt[:, 1:]

        n_words = gold.data.ne(Constants.PAD).sum().float()
        n_total_words += n_words
        batch_num += tgt.size(0)

        optimizer.zero_grad()
        pred = model(tgt, tgt_timestamp, graph, diffusion_graph)
        # backward
        loss, n_correct = get_performance(loss_func, pred, gold)
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate()

        # note keeping
        n_total_correct += n_correct
        total_loss += loss.item()
        print("Training batch ", i, " loss: ", loss.item(), " acc:", (n_correct.item()/len(pred)) )
        # print ("A Batch Time: ", str(time.time()-start_time))

    return total_loss/n_total_words, n_total_correct/n_total_words


def test_epoch(model, validation_data, graph, diffusion_graph, k_list=[10, 50, 100]):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0
    for i, batch in enumerate(validation_data):  #tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):
        print("Validation batch ", i)
        # prepare data
        tgt, tgt_timestamp = batch
        y_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()

        # forward
        pred = model(tgt, tgt_timestamp, graph, diffusion_graph)
        y_pred = pred.detach().cpu().numpy()

        scores_batch, scores_len = metric.compute_metric(y_pred, y_gold, k_list)
        n_total_words += scores_len
        for k in k_list:
            scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
            scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    return scores



parser = argparse.ArgumentParser()
parser.add_argument('-epoch', type=int, default=50) 

parser.add_argument('-batch_size', type=int, default=16)

parser.add_argument('-d_model', type=int, default=64)
# parser.add_argument('-d_inner_hid', type=int, default=64)

parser.add_argument('-n_warmup_steps', type=int, default=1000)

parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-embs_share_weight', action='store_true')
parser.add_argument('-proj_share_weight', action='store_true')

parser.add_argument('-log', default=None)
parser.add_argument('-save_path', default=root_path + "checkpoint/DiffusionPrediction.pt")
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

parser.add_argument('-no_cuda', action='store_true')

parser.add_argument('-network', type=bool, default=False) # use social network; need features or deepwalk embeddings as initial input
parser.add_argument('-pos_emb', type=bool, default=True)
parser.add_argument('-warmup', type=int, default=10) # warmup epochs
parser.add_argument('-notes', default='')
opt = parser.parse_args() 
opt.d_word_vec = opt.d_model
print(opt)

def train_model(DyHGCN, data_path):
    # ========= Preparing DataLoader =========#
    relation_graph = LoadRelationGraph(data_path)
    diffusion_graph = LoadDynamicHeteGraph(data_path)
    train_data = DataConstruct(data_path, data=0, load_dict=True, batch_size=opt.batch_size, cuda=False)
    valid_data = DataConstruct(data_path, data=1, batch_size=opt.batch_size, cuda=False) # torch.cuda.is_available()
    test_data = DataConstruct(data_path, data=2, batch_size=opt.batch_size, cuda=False)

    opt.user_size = train_data.user_size

        # ========= Preparing Model =========#
    model = DyHGCN(opt)
    loss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=Constants.PAD)
    
    params = model.parameters()
    optimizerAdam = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-09)
    optimizer = ScheduledOptim(optimizerAdam, opt.d_model, opt.n_warmup_steps)

    if torch.cuda.is_available():
        model = model.cuda()
        loss_func = loss_func.cuda()

    validation_history = 0.0
    for epoch_i in range(opt.epoch):
        print('\n[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, train_data, relation_graph, diffusion_graph, loss_func, optimizer)
        print('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu,
            elapse=(time.time() - start) / 60))

        if epoch_i >= 0: 
            start = time.time()
            scores = test_epoch(model, valid_data, relation_graph, diffusion_graph)
            print('  - ( Validation )) ')
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
            print("Validation use time: ", (time.time() - start) / 60, "min")

            print('  - (Test) ')
            scores = test_epoch(model, test_data, relation_graph, diffusion_graph)
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))

            if validation_history <= scores["hits@100"]:
                print("Best Validation hit@100:{} at Epoch:{}".format(scores["hits@100"], epoch_i))
                validation_history = scores["hits@100"]
                print("Save best model!!!")
                torch.save(model.state_dict(), opt.save_path)


def test_model(DyHGCN, data_path):
    relation_graph = LoadRelationGraph(data_path)
    diffusion_graph = LoadDynamicHeteGraph(data_path)

    test_data = DataConstruct(data_path, data=2, batch_size=opt.batch_size, cuda=torch.cuda.is_available())
    opt.user_size = test_data.user_size

    model = DyHGCN(opt)
    model.load_state_dict(torch.load(opt.save_path))
    model.cuda()

    scores = test_epoch(model, test_data, relation_graph, diffusion_graph)
    print('  - (Test) ')
    for metric in scores.keys():
        print(metric + ' ' + str(scores[metric]))


if __name__ == "__main__": 
    # data_path = "./data/twitter/"
    # data_path = "./data/douban/"
    data_path = "./data/memetracker/"
    model = DyHGCN_H  # DyHGCN_H
    train_model(model, data_path)
    test_model(model, data_path)



