# from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
# import transformers
# import torch
import os
import fire
import math
import json
import pandas as pd
import numpy as np
    
from tqdm import tqdm
def gao(path, item_path, result_data, group_title_path, category):
    import json
    with open(group_title_path, "r") as input_file:
        group_title = json.load(input_file)
    if category == 'CDs_and_Vinyl':
        gh_genre = [0.42919059700842693, 0.17071915707144072, 0.11809153627846408, 0.09089901560700277, 0.07123886274308015, 0.05642744900907605, 0.04134688077643157, 0.022086501506077747]
    elif category == 'Video_Games':
        gh_genre = [0.48097812357904307, 0.17252196598430114, 0.10991029886599027, 0.07947254079725155, 0.06112813069818072, 0.0467336937301066, 0.033188960284398064, 0.01606628606072854]
    elif category == 'Movies_and_TV':
        gh_genre = [0.3926707980937671, 0.17973591516133286, 0.1274746180941438, 0.09763792876113696, 0.07813294656143457, 0.060718792970295164, 0.043097440147676545, 0.02053156021021304]
    
    if type(path) != list:
        path = [path]
    if item_path.endswith(".txt"):
        item_path = item_path[:-4]
    CC=0
        
    
    f = open(f"{item_path}.txt", 'r')
    items = f.readlines()
    item_names = [_.split('\t')[0].strip("\"").strip(" ").strip('\n').strip('\"') for _ in items]
    item_ids = [_ for _ in range(len(item_names))]
    item_dict = dict()
    for i in range(len(item_names)):
        if item_names[i] not in item_dict:
            item_dict[item_names[i]] = [item_ids[i]]
        else:   
            item_dict[item_names[i]].append(item_ids[i])
    
    ALLNDCG = np.zeros(5) # 1 3 5 10 20
    ALLHR = np.zeros(5)

    result_dict = dict()
    topk_list = [1, 3, 5, 10, 20]
    for p in path:
        result_dict[p] = {
            "NDCG": [],
            "HR": [],
        }
        f = open(p, 'r')
        import json
        test_data = json.load(f)
        f.close()
        
        text = [ [_.strip(": \n").strip(" \n").strip("\"").strip(" ") for _ in sample["predict"]] for sample in test_data]

        def process_string(input_string):
            stripped_string = input_string.rstrip('\n')
            lst = stripped_string.split()
            return lst
        item_set = {}
        group_num = {}
        word_set = {}
        for index, sample in tqdm(enumerate(text)):

                if type(test_data[index]['output']) == list:
                    target_item = test_data[index]['output'][0].strip("\"").strip(" ")
                else:
                    target_item = test_data[index]['output'].strip(" \n")
                minID = 1000000
                # rank = dist.argsort(dim = -1)
                for i in range(len(sample)):
                    if sample[i] not in item_set:
                        item_set[sample[i]] = 0
                    item_set[sample[i]] += 1
                    i_word = process_string(sample[i])
                    for w in i_word:
                        if w not in word_set:
                            word_set[w] = 0
                        word_set[w] += 1
                    if sample[i] not in item_dict:
                        CC += 1
                    else:
                        flag=0
                        for j in group_title.keys():
                            if sample[i] in group_title[j]:
                                flag=1
                                if j in group_num:
                                    group_num[j] += 1
                                else:
                                    group_num[j] = 1
                        if flag==0:
                            group_num['7'] = 1
                    if sample[i] == target_item:
                        minID = i

                for index, topk in enumerate(topk_list):
                    if minID < topk:
                        ALLNDCG[index] = ALLNDCG[index] + (1 / math.log(minID + 2))
                        ALLHR[index] = ALLHR[index] + 1
        
        def calculate_entropy(word_counts):
            total_words = sum(word_counts.values())
            entropy = 0.0
            for count in word_counts.values():
                probability = count / total_words
                entropy += -probability * math.log2(probability)
            return entropy
        
        entropy = calculate_entropy(word_set)
        
        mean_NDCG = ALLNDCG / len(text) / (1.0 / math.log(2))
        mean_HR = ALLHR / len(text)

        sorted_items = sorted(group_num.items(), key=lambda item: int(item[0]))
        group_num = dict(sorted_items)
        total_sum = sum(group_num.values())
        group_prob = [v/total_sum for k,v in group_num.items()]
        dis_genre = [group_prob[i]-gh_genre[i] for i in range(len(gh_genre))]
        DGU_genre = max(dis_genre)-min(dis_genre)
        dis_abs_genre = [abs(x) for x in dis_genre]
        MGU_genre = sum(dis_abs_genre) / len(dis_genre)
        with open(result_data, 'w') as file:
            file.write(f"NDCG: {mean_NDCG}\n")
            file.write(f"HR: {mean_HR}\n")
            file.write(f"CC: {CC}\n")
            file.write(f"DGU: {DGU_genre}\n")
            file.write(f"MGU: {MGU_genre}\n")
            file.write(f"H-word: {entropy}\n")
            file.write(f"Title_num: {len(item_set)}\n")
            file.write(f"TTR: {len(word_set)/sum(word_set.values())}\n")
            file.write(f"Group prob after train: {group_prob}\n")
            file.write(f"Group prob in dataset: {gh_genre}\n")

if __name__=='__main__':
    fire.Fire(gao)
