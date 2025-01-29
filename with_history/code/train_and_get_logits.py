import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import os
import math
import logging
import time as Time
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from SASRecModules_ori import *
import random
import json
import copy
import numpy as np 

logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='games',
                        help='games, movie')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--num_filters', type=int, default=16,
                        help='num_filters')
    parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]',
                        help='Specify the filter_size')
    parser.add_argument('--r_click', type=float, default=0.2,
                        help='reward for the click behavior.')
    parser.add_argument('--r_buy', type=float, default=1.0,
                        help='reward for the purchase behavior.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--cuda', type=int, default=7,
                        help='cuda device.')
    parser.add_argument('--l2_decay', type=float, default=1e-5,
                        help='l2 loss reg coef.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='dro alpha.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='for robust radius')
    parser.add_argument("--model", type=str, default="SASRec",
                        help='the model name, GRU, Caser, SASRec')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--descri', type=str, default='',
                        help='description of the work.')
    parser.add_argument("--early_stop", type=int, default=20,
                        help='the epoch for early stop')
    parser.add_argument("--eval_num", type=int, default=1,
                        help='evaluate every eval_num epoch' )
    parser.add_argument("--seed", type=int, default=1,
                        help="the random seed")
    parser.add_argument("--result_json_path", type=str, default="./result_temp/temp.json")
    parser.add_argument("--group_ids", type=str, default="./result_temp/temp.json")
    parser.add_argument("--info_file", type=str, default="./result_temp/temp.json")
    parser.add_argument("--category", type=str, default="Steam")
    parser.add_argument("--sample", type=int, default = 65536)
    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GRU(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, gru_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.state_size = state_size
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )
        self.s_fc = nn.Linear(self.hidden_size, self.item_num)

    def forward(self, states, len_states):
        # Supervised Head
        emb = self.item_embeddings(states)
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(emb, len_states, batch_first=True, enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)
        return supervised_output

    def forward_eval(self, states, len_states):
        # Supervised Head
        emb = self.item_embeddings(states)
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(emb, len_states, batch_first=True, enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)

        return supervised_output


class Caser(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, num_filters, filter_sizes,
                 dropout_rate):
        super(Caser, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.filter_sizes = eval(filter_sizes)
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )

        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)

        # Horizontal Convolutional Layers
        self.horizontal_cnn = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (i, self.hidden_size)) for i in self.filter_sizes])
        # Initialize weights and biases
        for cnn in self.horizontal_cnn:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        # Vertical Convolutional Layer
        self.vertical_cnn = nn.Conv2d(1, 1, (self.state_size, 1))
        nn.init.xavier_normal_(self.vertical_cnn.weight)
        nn.init.constant_(self.vertical_cnn.bias, 0.1)

        # Fully Connected Layer
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        final_dim = self.hidden_size + self.num_filters_total
        self.s_fc = nn.Linear(final_dim, item_num)

        # dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, states, len_states):
        input_emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)

        return supervised_output

    def forward_eval(self, states, len_states):
        input_emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)
        
        return supervised_output

class SASRec(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1):
        super(SASRec, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)
        # self.ac_func = nn.ReLU()

    def forward(self, states, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to('cuda'))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        # state_hidden = extract_axis_1(ff_out, len_states - 1)
        indices = (len_states -1 ).view(-1, 1, 1).repeat(1, 1, self.hidden_size)
        state_hidden = torch.gather(ff_out, 1, indices)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output

    def forward_eval(self, states, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        # state_hidden = extract_axis_1(ff_out, len_states - 1)
        indices = (len_states -1 ).view(-1, 1, 1).repeat(1, 1, self.hidden_size)
        state_hidden = torch.gather(ff_out, 1, indices)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output


def evaluate_games(model, test_data, device, topk, model_type, group_ids_path, category, info_file):
    with open(group_ids_path, "r") as input_file:
        group_ids = json.load(input_file)

    def process_string(input_string):
        stripped_string = input_string.rstrip('\n')
        lst = stripped_string.split()
        return lst
        
    with open(info_file, 'r') as f:
        info = f.readlines()
        title = [_.split('\t')[0].strip(' ') for _ in info]
        id = [_.split('\t')[1].strip(' ').strip('\n') for _ in info]
    id2title = dict(zip(id, title))
    
    if category == 'Steam':
        gh_genre = [0.6767917454557378, 0.1286274852714534, 0.06988818327886412, 0.05437812196001705, 0.03566548984030867, 0.019368448665959843, 0.009793527090688498, 0.005486998436970565]
    elif category == 'CDs_and_Vinyl':
        gh_genre = [0.42919059700842693, 0.17071915707144072, 0.11809153627846408, 0.09089901560700277, 0.07123886274308015, 0.05642744900907605, 0.04134688077643157, 0.022086501506077747]
    elif category == 'Musical_Instruments':
        gh_genre = [0.5490769565754706, 0.15251843438653068, 0.09438170508512479, 0.06851177270239574, 0.052534323562470946, 0.040511513767676706, 0.02864759573228504, 0.013817698188045454]
    elif category == 'Video_Games':
        gh_genre = [0.48097812357904307, 0.17252196598430114, 0.10991029886599027, 0.07947254079725155, 0.06112813069818072, 0.0467336937301066, 0.033188960284398064, 0.01606628606072854]
    elif category == 'Movielens':
        gh_genre = [0.39007213606391483, 0.21822188611212834, 0.13411587275102982, 0.09193718593509272, 0.0641943794976695, 0.046757344162049366, 0.034269505882047024, 0.020431689596068378]
    elif category == 'GoodReads':
        gh_genre = [0.4275590595083439, 0.1641316286875603, 0.11539533768601584, 0.0886993805512035, 0.07281171767408406, 0.05994881602203476, 0.04655623268080266, 0.024897827189955004]
    elif category == 'Movies_and_TV':
        gh_genre = [0.3926707980937671, 0.17973591516133286, 0.1274746180941438, 0.09763792876113696, 0.07813294656143457, 0.060718792970295164, 0.043097440147676545, 0.02053156021021304]
    elif category == 'Books':
        gh_genre = [0.32741057800783063, 0.1919337713186011, 0.1454310173203265, 0.11450660295971862, 0.09084876235981153, 0.06813657176985866, 0.043566261862101004, 0.01816643440175194]
    
    
    def calculate_hit_games_cuda(prediction, topk_list, target, hit_all, ndcg_all, group_num, word_set):
        rank_list = (prediction.shape[1] - 1 - torch.argsort(torch.argsort(prediction)))
        indices = torch.argmax((rank_list == 0).int(), dim=1)
        top0_indices = indices.tolist()
        for i in top0_indices:
            flag=0
            for k,v in group_ids.items():
                if id2title[str(i)] in v:
                    flag=1
                    group_num[k] += 1
                    break
            if flag==0:
                group_num['7'] += 1
            i_word = process_string(id2title[str(i)])
            for w in i_word:
                if w not in word_set:
                    word_set[w] = 0
                word_set[w] += 1
            
        target_rank = torch.gather(rank_list, 1, target.view(-1, 1)).view(-1).clone()
        ndcg_temp_full = 1 / torch.log2(target_rank + 2)
        for i, top_k in enumerate(topk_list):
            mask = (target_rank < top_k)
            mask = mask.float()
            recall_temp = mask.sum()
            ndcg_temp = (ndcg_temp_full * mask).sum()
            hit_all[i] += recall_temp.cpu().item()
            ndcg_all[i] += ndcg_temp.cpu().item()
        return hit_all, ndcg_all, group_num, word_set


    eval_seqs=pd.read_pickle(os.path.join(data_directory, test_data))
    batch_size=1024
    hit_all = []
    ndcg_all = []
    for i in topk:
        hit_all.append(0)
        ndcg_all.append(0)
    total_samples = len(eval_seqs)
    total_batch_num = int(total_samples/batch_size) + (total_samples > batch_size * int(total_samples/batch_size))
    best_prediction = []
    word_set = {}
    group_num = {}
    for k in range(len(gh_genre)):
        group_num[str(k)] = 0
    for i in range(total_batch_num):
        begin = i * batch_size
        end = (i + 1) * batch_size
        if end > total_samples:
            batch = eval_seqs[begin:]
        else:
            batch = eval_seqs[begin:end]

        seq = list(batch['seq'].tolist())
        len_seq = list(batch['len_seq'])
        target=list(batch['next'])

        seq = torch.LongTensor(seq)

        seq = seq.to(device)

        target = torch.LongTensor(target).to(device)

        _ = model.eval()
        with torch.no_grad():
            if model_type != "GRU":
                prediction = model.forward_eval(seq, torch.tensor(np.array(len_seq)).to(device))
            else:
                prediction = model.forward_eval(seq, torch.tensor(np.array(len_seq)).to("cpu"))
            best_prediction.append(prediction)
        hit_all, ndcg_all, group_num, word_set = calculate_hit_games_cuda(prediction,topk, target, hit_all, ndcg_all, group_num, word_set)
    best_prediction = torch.cat(best_prediction)
    print('#############################################################')
    hr_list = []
    ndcg_list = []
    for i in range(len(topk)):
        hr_purchase=hit_all[i]/len(eval_seqs)
        ng_purchase=ndcg_all[i]/len(eval_seqs)

        hr_list.append(hr_purchase)
        try:
            ndcg_list.append(ng_purchase)
        except:
            if ng_purchase == 0:
                ndcg_list.append(0)
            else:
                return "error"

    ndcg_last = ndcg_list[-1]

    def calculate_entropy(word_counts):
        total_words = sum(word_counts.values())
        entropy = 0.0
        for count in word_counts.values():
            probability = count / total_words
            entropy += -probability * math.log2(probability)
        return entropy

    str1 = ''
    str2 = ''
    for i in range(len(topk)):
        str1 += 'hr@{}\tndcg@{}'.format(topk[i], topk[i])
        str2 += '{:.6f}\t{:.6f}\t'.format(hr_list[i], ndcg_list[i])

    print(str1)
    print(str2)
    entropy = calculate_entropy(word_set)
    ttr = len(word_set)/sum(word_set.values())

    sorted_items = sorted(group_num.items(), key=lambda item: int(item[0]))
    group_num = dict(sorted_items)
    total_sum = sum(group_num.values())
    group_prob = [v/total_sum for k,v in group_num.items()]
    dis_genre = [group_prob[i]-gh_genre[i] for i in range(len(gh_genre))]
    DGU_genre = max(dis_genre)-min(dis_genre)
    dis_abs_genre = [abs(x) for x in dis_genre]
    MGU_genre = sum(dis_abs_genre) / len(dis_genre)
    print(f'DGU: {DGU_genre}')
    print(f'MGU: {MGU_genre}')
    print(f'Entropy: {entropy}')
    print(f'TTR: {ttr}')
    print('#############################################################')


    return ndcg_last, hr_list, ndcg_list, best_prediction.cpu().numpy(), DGU_genre, MGU_genre, group_prob, entropy, ttr


def calcu_propensity_score(buffer):
    items = list(buffer['next'])
    freq = Counter(items)
    for i in range(item_num):
        if i not in freq.keys():
            freq[i] = 0
    pop = [freq[i] for i in range(item_num)]
    pop = np.array(pop)
    ps = pop + 1
    ps = ps / np.sum(ps)
    ps = np.power(ps, 0.05)
    return ps

class RecDataset(Dataset):
    def __init__(self, data_df):
        self.data = data_df

    def __getitem__(self, i):
        temp = self.data.iloc[i]
        seq = torch.tensor(temp['seq'])
        len_seq = torch.tensor(temp['len_seq'])
        next = torch.tensor(temp['next'])
        return seq, len_seq, next

    def __len__(self):
        return len(self.data)

def main(topk=[1,3,5,10,20]):
    if args.model=='SASRec':
        model = SASRec(args.hidden_factor,item_num, seq_size, args.dropout_rate, device)
    if args.model=="GRU":
        model = GRU(args.hidden_factor,item_num, seq_size)
    if args.model=="Caser":
        model = Caser(args.hidden_factor,item_num, seq_size, args.num_filters, args.filter_sizes, args.dropout_rate)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    bce_loss = nn.BCEWithLogitsLoss()

    model.to(device)

    train_data_org = pd.read_pickle(os.path.join(data_directory, 'train.pkl'))
    
    if args.sample == -1:
        args.sample = len(train_data_org)
    if len(train_data_org)<args.sample:
        args.sample = len(train_data_org)

    train_data = train_data_org.sample(n=args.sample ,random_state=args.seed)
    

    train_dataset = RecDataset(train_data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    ps = calcu_propensity_score(train_data)
    ps = torch.tensor(ps)
    ps = ps.to(device)

    total_step=0
    ndcg_max = -1
    best_epoch = 0
    early_stop = 0

    num_rows=train_data.shape[0]
    num_batches=int(num_rows/args.batch_size) + (int(num_rows/args.batch_size) * args.batch_size != num_rows)
    for i in range(args.epoch):
        # for j in tqdm(range(num_batches)):
        for j, (seq, len_seq, target) in tqdm(enumerate(train_loader)):
            target_neg = []
            for index in range(len(len_seq)):
                neg=np.random.randint(item_num)
                while neg==target[index]:
                    neg = np.random.randint(item_num)
                target_neg.append(neg)
            optimizer.zero_grad()
            seq = torch.LongTensor(seq)
            len_seq = torch.LongTensor(len_seq)
            target = torch.LongTensor(target)
            target_neg = torch.LongTensor(target_neg)
            seq = seq.to(device)
            target = target.to(device)
            len_seq = len_seq.to(device)
            target_neg = target_neg.to(device)

            if args.model=="GRU":
                len_seq = len_seq.cpu()

            model_output = model.forward(seq, len_seq)


            target = target.view((-1, 1))
            target_neg = target_neg.view((-1, 1))

            pos_scores = torch.gather(model_output, 1, target)
            neg_scores = torch.gather(model_output, 1, target_neg)

            pos_labels = torch.ones((len(len_seq), 1))
            neg_labels = torch.zeros((len(len_seq), 1))

            scores = torch.cat((pos_scores, neg_scores), 0)
            labels = torch.cat((pos_labels, neg_labels), 0)
            labels = labels.to(device)

            loss = bce_loss(scores, labels)


            pos_scores_dro = torch.gather(torch.mul(model_output * model_output, ps), 1, target)
            pos_scores_dro = torch.squeeze(pos_scores_dro)
            pos_loss_dro = torch.gather(torch.mul((model_output - 1) * (model_output - 1), ps), 1, target)
            pos_loss_dro = torch.squeeze(pos_loss_dro)

            inner_dro = torch.sum(torch.exp((torch.mul(model_output * model_output, ps) / args.beta)), 1) - torch.exp((pos_scores_dro / args.beta)) + torch.exp((pos_loss_dro / args.beta)) 


            loss_dro = torch.log(inner_dro + 1e-24)
            if args.alpha == 0.0:
                loss_all = loss
            else:
                loss_all = loss + args.alpha * torch.mean(loss_dro)
            loss_all.backward()
            optimizer.step()

            if True:

                total_step+=1


                if total_step % (num_batches * args.eval_num) == 0:
                        print('VAL PHRASE:')
                        ndcg_last, val_hr, val_ndcg, valid_predict_score, _, _, _, _, _ = evaluate_games(model, 'eval.pkl', device, topk, args.model, args.group_ids, args.category, args.info_file)
                        # ndcg_last, val_hr, val_ndcg = evaluate_games_old(model, 'eval.pkl', device, topk)
                        print('TEST PHRASE:')
                        _, test_hr, test_ndcg, predict_score, DGU_genre, MGU_genre, group_prob, entropy, ttr = evaluate_games(model, 'test.pkl', device, topk, args.model, args.group_ids, args.category, args.info_file)
                        print('TRAIN PHRASE:')
                        _, train_hr, train_ndcg, train_predict_score, _, _, _, _, _ = evaluate_games(model, 'train.pkl', device, topk, args.model, args.group_ids, args.category, args.info_file)

                        model = model.train()

                        if ndcg_last > ndcg_max:

                            ndcg_max = ndcg_last
                            best_epoch = i
                            early_stop = 0
                            best_hr = test_hr
                            best_ndcg = test_ndcg
                            best_model = copy.deepcopy(model)
                            best_predict = predict_score
                            valid_best_predict = valid_predict_score
                            train_best_predict = train_predict_score
                            best_DGU_genre = DGU_genre
                            best_MGU_genre = MGU_genre
                            best_group_prob = group_prob
                            beat_entropy = entropy
                            beat_ttr = ttr
                        else:
                            early_stop += 1
                            if early_stop > args.early_stop:
                                return model, best_ndcg, best_hr, best_predict, valid_best_predict, train_best_predict, best_DGU_genre, best_MGU_genre, best_group_prob, beat_entropy, beat_ttr
                        
                        print('BEST EPOCH:{}'.format(best_epoch))
                        print('EARLY STOP:{}'.format(early_stop))
                        print("best hr:")
                        print(best_hr)
                        print("best ndcg")
                        print(best_ndcg)
                        print("best_DGU_genre")
                        print(best_DGU_genre)
                        print("best_MGU_genre")
                        print(best_MGU_genre)
    return best_model, best_ndcg, best_hr, best_predict, valid_best_predict, train_best_predict, best_DGU_genre, best_MGU_genre, best_group_prob, beat_entropy, beat_ttr
    


if __name__ == '__main__':
    topk=[1,3,5,10,20]
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    setup_seed(args.seed)

    data_directory = args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.pkl'))  # read data statistics, includeing seq_size and item_num
    seq_size = data_statis['seq_size'][0]  # the length of history to define the seq
    item_num = data_statis['item_num'][0]  # total number of items

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_model, test_ndcg, test_hr, best_predict, valid_best_predict, train_best_predict, best_DGU_genre, best_MGU_genre, best_group_prob, beat_entropy, beat_ttr = main(topk)

    result_dict = {}
    result_dict["NDCG"] = {}
    result_dict["HR"] = {}
    for i,k in enumerate(topk):
        result_dict["NDCG"][k] = test_ndcg[i]
        result_dict["HR"][k] = test_hr[i]
    result_dict["DGU"] = best_DGU_genre
    result_dict["MGU"] = best_MGU_genre
    result_dict["P"] = best_group_prob
    result_dict["Entropy"] = beat_entropy
    result_dict["TTR"] = beat_ttr

    result_folder = ""
    for path_name in args.result_json_path.split("/")[:-1]:
        result_folder += path_name + "/"

    os.makedirs(result_folder, exist_ok=True)
    with open(args.result_json_path ,'w',encoding='utf-8') as f:
        json.dump(result_dict, f,ensure_ascii=False, indent=1)
    torch.save(best_model, result_folder + "/best" +  str(args.lr) + "_" + str(args.l2_decay) + "_" +str(args.seed) + ".pt")
    np.save(result_folder + "/test_best" +  str(args.lr) + "_" + str(args.l2_decay) + "_" +str(args.seed) + ".npy", best_predict)
    np.save(result_folder + "/valid_best" +  str(args.lr) + "_" + str(args.l2_decay) + "_" +str(args.seed) + ".npy", valid_best_predict)
    np.save(result_folder + "/train_best" +  str(args.lr) + "_" + str(args.l2_decay) + "_" +str(args.seed) + ".npy", train_best_predict)

