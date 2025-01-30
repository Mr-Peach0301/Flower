import os
from trl import (
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    PPOTrainer,
)
from torch.nn.utils.rnn import pad_sequence
import sys
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import fire
import math
import scipy.stats
from transformers import  LogitsProcessorList, TemperatureLogitsWarper, GenerationConfig
from LogitProcesser import CFEnhancedLogitsProcessor
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from trl.core import LengthSampler
from accelerate import Accelerator
from dataset import D3Dataset

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

def gao(path, item_path, result_data, group_title_path, category):
    import json
    with open(group_title_path, "r") as input_file:
        group_title = json.load(input_file)
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
        
        text = [ [_.strip(": \n").strip("\"").strip(" ") for _ in sample["predict"]] for sample in test_data]
        
        def process_string(input_string):
            stripped_string = input_string.rstrip('\n')
            lst = stripped_string.split()
            return lst
        
        group_num = {}
        item_set = {}
        word_set = {}
        for k in range(len(gh_genre)):
            group_num[str(k)] = 0
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
        with open(result_data+'/'+'result.txt', 'w') as file:
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

def get_tensors_from_dataframe(df_train, tokenizer):
    input_ids = []
    target_ids = []
    index = []
    for i in range(len(df_train)):
        input_ids.append(tokenizer(df_train[i]['prompt'], return_tensors='pt')["input_ids"][0])
        target_ids.append(tokenizer(df_train[i]['chosen'], return_tensors='pt')["input_ids"][0])
        index.append(torch.tensor(i))
    max_len_input = max(len(seq) for seq in input_ids)
    max_len_target = max(len(seq) for seq in target_ids)
    input_ids = torch.stack([
        F.pad(seq, (max_len_input - len(seq), 0), value=tokenizer.eos_token_id)
        for seq in input_ids
    ])

    target_ids = torch.stack([
        F.pad(seq, (max_len_target - len(seq), 0), value=tokenizer.eos_token_id)
        for seq in target_ids
    ])

    index = torch.stack(index)

    num_samples = input_ids.size(0)
    
    indices = torch.randperm(num_samples)[:1024]
    input_ids = input_ids[indices]
    target_ids = target_ids[indices]
    index = index[indices]

    return input_ids, target_ids, index
 
def on_epoch_end(model, path, category, info_file, tokenizer, logits_file, test_data_path, batch_size, result_data, group_title_path):
    with torch.no_grad():
        category_dict = {"Video_Games": "video games", "Steam": "games", "CDs_and_Vinyl": "musics", 'Musical_Instruments': 'musical instruments', 'GoodReads': 'books', 'Movielens': 'movies', 'Movies_and_TV': 'movies and TV', 'Books': 'books'}
        o_category = category
        category = category_dict[category]
        with open(info_file, 'r') as f:
            info = f.readlines()
            info = [_.split('\t')[0].strip(' ') + "\n" for _ in info]
            item_name = info
            info = [f'''### Response: 
{_}''' for _ in info]
        device = next(model.parameters()).device
        model.eval()
        prefixID = [tokenizer(_).input_ids for _ in info]
    
        hash_dict = dict()
        sasrec_dict = dict()
        for index, ID in enumerate(prefixID):
            ID.append(tokenizer.eos_token_id)
            for i in range(4, len(ID)):
                if i == 4:
                    hash_number = get_hash(ID[:i])
                else:
                    hash_number = get_hash(ID[4:i])
                if hash_number not in hash_dict:
                    hash_dict[hash_number] = set()
                    sasrec_dict[hash_number] = set()
                hash_dict[hash_number].add(ID[i])
                sasrec_dict[hash_number].add(index)
            hash_number = get_hash(ID[4:])
            if hash_number not in sasrec_dict:
                sasrec_dict[hash_number] = set()
            sasrec_dict[hash_number].add(index)

        for key in hash_dict.keys():
            hash_dict[key] = list(hash_dict[key])
        for key in sasrec_dict.keys():
            sasrec_dict[key] = list(sasrec_dict[key])
    
        def prefix_allowed_tokens_fn(batch_id, input_ids):
            hash_number = get_hash(input_ids)
            if hash_number in hash_dict:
                return hash_dict[hash_number]
            return []

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        val_dataset=D3Dataset(train_file=test_data_path, tokenizer=tokenizer,max_len=2560, category=category, test=True,K=0, seed=0)
            
        if logits_file is not None:
            if not logits_file.endswith(".npy"):
                logits_file = None
    
        if logits_file is not None:
            logits = np.load(logits_file)
            sasrec_logits = torch.tensor(logits).softmax(dim = -1)
        encodings = [val_dataset.__getitem__(i) for i in range(len(val_dataset))]
        test_data = val_dataset.get_all()

        model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id

        model.eval()

        def evaluate(
            encodings,
            cf_logits,
            temperature=1.0,
            num_beams=10,
            max_new_tokens=100,
            guidance_scale=1.0,
            length_penalty=1.0,
            **kwargs,
        ):
            maxLen = max([len(_["input_ids"]) for _ in encodings])

            padding_encodings = {"input_ids": []}

            for  _ in encodings:
                L = len(_["input_ids"])
                padding_encodings["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])
            
            generation_config = GenerationConfig(
                num_beams=num_beams,
                length_penalty=length_penalty,
                num_return_sequences=num_beams,
                pad_token_id = model.config.pad_token_id,
                eos_token_id = model.config.eos_token_id,
                max_new_tokens = max_new_tokens,
                **kwargs
            )
            with torch.no_grad():
                ccc = CFEnhancedLogitsProcessor(
                    guidance_scale=guidance_scale,
                    cf_logits=cf_logits,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    cf_dict=sasrec_dict,
                    unconditional_ids=None,
                    model=model,
                    tokenizer=tokenizer,
                    num_beams=num_beams
                )
                logits_processor = LogitsProcessorList([TemperatureLogitsWarper(temperature=float(temperature)), ccc])
                    
                generation_output = model.generate(
                    torch.tensor(padding_encodings["input_ids"]).to(device),
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    logits_processor=logits_processor,
                )
               
                
            s = generation_output.sequences[:, L:]
            sequence_scores = [[0 for i in range(len(generation_output.scores))] for _ in range(num_beams)]
            for i in range(num_beams):
                for j in range(L, len(generation_output.sequences[i])):
                    beam_index = generation_output.beam_indices[i][j - L]
                    if beam_index != -1:
                        sequence_scores[i][j - L] = generation_output.scores[j - L][beam_index][generation_output.sequences[i][j]].item()
            scores = generation_output.sequences_scores.tolist()
            output = tokenizer.batch_decode(s, skip_special_tokens=True)
            output = [_.split("Response:")[-1] for _ in output]
            real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
            real_scores = [scores[i * num_beams: (i + 1) * num_beams] for i in range(len(scores) // num_beams)]
            return real_outputs, real_scores, sequence_scores
                
    
        model = model.to(device)

        from tqdm import tqdm
        outputs = []
        new_encodings = []
        BLOCK = (len(encodings) + batch_size - 1) // batch_size
        for i in range(BLOCK):
            new_encodings.append(encodings[i * batch_size: (i + 1) * batch_size])
        Flg=True
        scores = []
        seq_scores = []
        import random
        for idx, encodings in enumerate(tqdm(new_encodings)):
            if logits_file is not None:
                output, score, seq_score = evaluate(encodings, sasrec_logits[idx].to(device), temperature=1.0, guidance_scale=1.0, length_penalty=1.0)
            else:
                output, score, seq_score = evaluate(encodings, cf_logits=None, temperature=1.0, guidance_scale=1.0, length_penalty=1.0)
            outputs = outputs + output
            scores = scores + score
            seq_scores.append(seq_score)

        for i, test in enumerate(test_data):
            test["predict"] = outputs[i]

        for i in range(len(test_data)):
            if 'dedup' in test_data[i]:
                test_data[i].pop('dedup')  
    
        with open(path +'.json', 'w') as f:
            json.dump(test_data, f, indent=4)

        gao(path + '.json', info_file, result_data, group_title_path, o_category)

def train(
    base_model: str = "",
    learning_rate: float = 1e-7,
    steps: int = 512,
    batch_size: int = 4,
    ppo_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    early_stopping: bool = False,
    target_kl: float = 0.1,
    init_kl_coef: float = 0.9,
    kl_penalty: str = "abs",
    adap_kl_ctrl: bool = True,
    mini_batch_size: int = 2,
    optimize_cuda_cache: bool = True,
    seed: int = 42,
    train_file: str = None,
    num_samples_per_batch: int = 32,
    title_distri: str = None,
    category: str = 'movie',
    real_token_distri: str = None,
    save_path: str = None,
    test_data_path: str = None,
    info_path: str = None,
    group_title_path: str = None,
    sasrec_logits_path: str = None,

):
    path = f'./temp/{category}_base/sft_ppo_1024/final_result'
    result_data = f'./temp/{category}_base/sft_ppo_1024'
    if not os.path.exists(result_data):
            os.makedirs(result_data)

    with open(info_path, 'r') as f:
        info = f.readlines()
        title = [_.split('\t')[0].strip(' ') for _ in info]
        ids = [_ for _ in range(len(title))]
    title2id = dict(zip(title, ids))

    with open(title_distri, "r") as input_file:
        t2i = json.load(input_file)

    def normalize_dict_values(d):
        values = list(d.values())
        min_val = min(values)
        max_val = max(values)

        if min_val == max_val:
            return {k: 0.5 for k in d}

        normalized_dict = {k: (v - min_val) / (max_val - min_val) for k, v in d.items()}

        return normalized_dict

    t2p = normalize_dict_values(t2i)

    min_non_zero = min([value for value in t2p.values() if value != 0])

    t2p = {key: (min_non_zero/2 if value == 0 else value) for key, value in t2p.items()}

    current_device = Accelerator().local_process_index

    inference_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model,
        device_map={"": current_device},
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, add_bos_token=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    config = PPOConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mini_batch_size=mini_batch_size,
        init_kl_coef=init_kl_coef,
        ppo_epochs=ppo_epochs,
        kl_penalty=kl_penalty,
        seed=seed
    )

    eos_string = tokenizer.decode(tokenizer.eos_token_id)

    with open(train_file, "r") as input_file:
        df_train = json.load(input_file)

    input_ids, target_ids, train_index = get_tensors_from_dataframe(df_train, tokenizer)
    train_dataset = TensorDataset(input_ids, target_ids, train_index)


    def collator(batch):
        input_ids, target_ids, index = zip(*batch)
        return {
            "input_ids": input_ids,
            "query": [s.replace(eos_string, "") for s in tokenizer.batch_decode(input_ids)],
            "target_ids": target_ids,
            "index": index,
        }


    ppo_trainer = PPOTrainer(
        config=config,
        model=inference_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
    )

    generation_kwargs = {
        "max_new_tokens": 100,
        "top_k": 0,
        "top_p": 1,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with open(info_path, 'r') as f:
        info = f.readlines()
        info = [_.split('\t')[0].strip(' ') + "\n" for _ in info]
        info = [f'''### Response: 
{_}''' for _ in info]
    
    prefixID = [tokenizer(_).input_ids for _ in info]

    hash_dict = dict()
    sasrec_dict = dict()
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(4, len(ID)):
            if i == 4:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[4:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
                sasrec_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])
            sasrec_dict[hash_number].add(index)
        hash_number = get_hash(ID[4:])
        if hash_number not in sasrec_dict:
            sasrec_dict[hash_number] = set()
        sasrec_dict[hash_number].add(index)

    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])
    for key in sasrec_dict.keys():
        sasrec_dict[key] = list(sasrec_dict[key])
    
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []

    def normalize_element(tensor, a, b):
        row_a = tensor[a, :]
        max_val, _ = torch.max(row_a, dim=0)
        min_val, _ = torch.min(row_a, dim=0)
        if max_val == min_val:
            raise ValueError("Cannot normalize when max and min values are the same.")
        element_b = tensor[a, b]
        normalized_value = (element_b - min_val) / (max_val - min_val)
        return normalized_value.item() 

    logits = np.load(sasrec_logits_path)
    sasrec_logits = torch.tensor(logits).softmax(dim = -1)

    inference_model.train()
    for idex, batch in enumerate(tqdm(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        train_i = batch['index']
        response_tensors = []
        for query in query_tensors:
            ccc = CFEnhancedLogitsProcessor(
                guidance_scale=1,
                cf_logits=None,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                cf_dict=sasrec_dict,
                unconditional_ids=None,
                model=inference_model,
                tokenizer=tokenizer,
                num_beams=1
            )
            logits_processor = LogitsProcessorList([TemperatureLogitsWarper(temperature=1.0), ccc])
            response = ppo_trainer.generate(query, **generation_kwargs, logits_processor=logits_processor)
            response_tensors.append(response.squeeze()[query.shape[0] :])
        batch["response"] = [r.clone() for r in response_tensors]
        pro = []
        
        for index, response in enumerate(response_tensors):
            decoded_text = tokenizer.decode(response, skip_special_tokens=True).strip('\n')
            if decoded_text in t2p:
                temp = normalize_element(sasrec_logits, train_i[index], title2id[decoded_text])
                if temp == 0:
                    pro.append(1e-20)
                else:
                    pro.append(temp)
            else:
                pro.append(0)
        rewards = []
        for index, response in enumerate(batch["response"]):
            rewards.append(torch.tensor(float(pro[index])).cuda())
        stats = ppo_trainer.step(list(query_tensors), response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
    on_epoch_end(model=inference_model, path=path, category=category, info_file=info_path, tokenizer=tokenizer,logits_file=None, test_data_path=test_data_path, batch_size=batch_size, result_data=result_data, group_title_path=group_title_path)

if __name__ == "__main__":
    fire.Fire(train)