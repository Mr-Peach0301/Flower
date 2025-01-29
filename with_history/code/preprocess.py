import pandas as pd
from tqdm import tqdm
import fire
import random
import pickle
import os
random.seed(1024)


def process_seq_data(data, max_len=10, max_item=11266, result_path="./", ST = 0, ED=1e14, istrain=False):
    seq_len = 0
    process_seq_data = []
    for seq_id in tqdm(range(len(data))):
        item = data.iloc[seq_id]
        history_item_id = eval(item['history_item_id'])
        if "history_timestamp" in item.keys():
            history_timestamp = eval(item['history_timestamp'])
            history_timestamp = [(int(_) - ST) // 864000 + 1 for _ in history_timestamp]
        else:
            history_timestamp = [0 for _ in range(len(history_item_id))]
        
        for i in range(len(history_timestamp)):
            if history_timestamp[i] > ED:
                history_timestamp[i] = 0
            if istrain:
                if random.random() < 0.15:
                    history_timestamp[i] = 0

        len_seq = len(history_item_id)
        assert len(history_item_id) <= max_len
        if len(history_item_id) < max_len:
            history_item_id =  history_item_id + [max_item] * (max_len - len_seq)
            history_timestamp = history_timestamp + [0] * (max_len - len_seq)
        if len_seq > seq_len:
            seq_len = len_seq
        next_item = item['item_id']
        process_seq_data.append([history_item_id, history_timestamp, len_seq, next_item])
    processed_df = pd.DataFrame(process_seq_data, columns=['seq', 'pos', 'len_seq', 'next'])
    processed_df.to_pickle(result_path)
    return seq_len

def get_items_set(data):
    temp_set = set()
    for index, row in data.iterrows():
        history_item_id = eval(row['history_item_id'])
        item_id = row['item_id']
        temp_set.add(item_id)
        for item in history_item_id:
            temp_set.add(item)
    return temp_set

def get_st_timestamp(data):
    st = None
    ed = None
    if "history_timestamp" not in data.columns:
        return 0, 1
    for index, row in data.iterrows():
        history_timestamp = eval(row["history_timestamp"])
        if st is None:
                st = int(history_timestamp[0])
                ed = int(history_timestamp[0])
        else:
                st = min(st, int(history_timestamp[0]))
                ed = max(ed, int(history_timestamp[0]))
    return st, ed

def gao(start_year, category, train_file, eval_file, test_file):
    data_train = pd.read_csv(train_file)
    ST, ED = get_st_timestamp(data_train)
    data_eval = pd.read_csv(eval_file)
    data_test = pd.read_csv(test_file)
    train_items = get_items_set(data_train)
    eval_items = get_items_set(data_eval)
    test_items = get_items_set(data_test)
    total_items = len(train_items | eval_items | test_items)
    max_items = max(train_items | eval_items | test_items) + 1
    if not os.path.exists(f'./cache/{category}_{start_year}'):
        os.makedirs(f'./cache/{category}_{start_year}')
    train_len = process_seq_data(data_train, max_item = max_items, result_path=f'./cache/{category}_{start_year}/train.pkl', ST=ST,ED=ED, istrain=True)
    eval_len = process_seq_data(data_eval, max_item = max_items, result_path=f'./cache/{category}_{start_year}/eval.pkl', ST=ST,ED=ED)
    test_len = process_seq_data(data_test, max_item = max_items, result_path=f'./cache/{category}_{start_year}/test.pkl', ST=ST,ED=ED)
    seq_len = max(train_len, eval_len, test_len)
    data_statis = {
        'seq_size': [seq_len],
        'item_num': [max_items]
    }
    print(seq_len)
    print(max_items)
    with open(f'./cache/{category}_{start_year}/data_statis.pkl', 'wb') as file:
        pickle.dump(data_statis, file)

    
    
    
if __name__ == '__main__':
    fire.Fire(gao)