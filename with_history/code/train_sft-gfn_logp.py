import os
import sys
from typing import List
import numpy as np 
import fire
import torch
import time
import json
import random
import transformers
from LogitProcesser import CFEnhancedLogitsProcessor
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, GenerationConfig
from torch.nn.utils.rnn import pad_sequence
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import math
import pandas as pd
from tqdm import tqdm
import warnings
from functools import partial
import numpy as np 
import fire
import transformers
from datasets import load_dataset, concatenate_datasets
from torch.optim.lr_scheduler import LambdaLR
"""
Unused imports:`
import torch.nn as nn
import bitsandbytes as bnb
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, LogitsProcessorList, TemperatureLogitsWarper, Trainer
from dataset import D3Dataset
from utils_logp_n import (
    FrozenModelSentenceGivenPrompt,
    generate_and_return_termination_logprob,
    modified_subtb_loss,
)

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

def gao(path, item_path, result_data, group_title_path, category, epoch):
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
        
        text = [ [_.strip(" \n").strip("\"").strip(" ") for _ in sample["predict"]] for sample in test_data]
        
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
        with open(result_data+'/'+str(epoch)+'.txt', 'w') as file:
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

class EvaluateCallback(TrainerCallback):
    def __init__(self, test_data_path, category, info_file, group_title_path, tokenizer, temperature, logits_file=None, title_token_distribution=None, possible_output=None, gfn_theta=0.9, train_theta=0.999, n=1):
        self.test_data_path = test_data_path
        self.category = category
        self.info_file = info_file
        self.batch_size = 1
        self.group_title_path = group_title_path
        self.tokenizer = tokenizer
        self.n = n
        self.path = f'./temp_sft&gfn_logp_{self.n}/{self.category}_base'
        self.result_data = f'./temp_sft&gfn_logp_{self.n}/{self.category}_base'
        self.temperature = temperature
        self.logits_file = logits_file
        try:
            self.end_of_sentence_token_id = self.tokenizer.encode(
                "A sentence\n", add_special_tokens=False
            )[-1]
        except:
            self.end_of_sentence_token_id = self.tokenizer.convert_tokens_to_ids("\n")
        with open(possible_output, 'r', encoding='utf-8') as file:
            temp = json.load(file)
        self.possible_output = list(temp.keys())
        self.title_token_distribution = title_token_distribution
        self.gfn_theta = gfn_theta
        self.train_theta = train_theta
        self.result_data = os.path.join(self.result_data, str(self.train_theta)+'_'+str(self.gfn_theta))
        if not os.path.exists(self.result_data):
            os.makedirs(self.result_data)
        self.path = os.path.join(self.result_data, 'final_result')

    def perform_evaluation(self, args, state, control, epoch_or_step, **kwargs):
        with torch.no_grad():
            category_dict = {"Video_Games": "video games", "Steam": "games", "CDs_and_Vinyl": "musics", 'Musical_Instruments': 'musical instruments', 'GoodReads': 'books', 'Movielens': 'movies'}
            category = category_dict[self.category]
            with open(self.info_file, 'r') as f:
                info = f.readlines()
                info = [_.split('\t')[0].strip(' ') + "\n" for _ in info]
                item_name = info
                info = [f'''### Response: 
{_}''' for _ in info]
            model = kwargs["model"]
            tokenizer = self.tokenizer
            device = next(model.parameters()).device
            model.eval()
            prefixID = [tokenizer(_).input_ids for _ in info]

            epoch = state.epoch
    
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
            val_dataset=D3Dataset(train_file=self.test_data_path, tokenizer=tokenizer,max_len=2560, category=category, test=True,K=0, seed=0)
            
            if self.logits_file is not None:
                if not self.logits_file.endswith(".npy"):
                    self.logits_file = None
    
            if self.logits_file is not None:
                logits = np.load(self.logits_file)
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
                    max_new_tokens=64,
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
            BLOCK = (len(encodings) + self.batch_size - 1) // self.batch_size
            for i in range(BLOCK):
                new_encodings.append(encodings[i * self.batch_size: (i + 1) * self.batch_size])
            Flg=True
            scores = []
            seq_scores = []
            import random
            for idx, encodings in enumerate(tqdm(new_encodings)):
                if self.logits_file is not None:
                    output, score, seq_score = evaluate(encodings, sasrec_logits[idx].to(device), temperature=self.temperature, guidance_scale=1.0, length_penalty=1.0)
                else:
                    output, score, seq_score = evaluate(encodings, cf_logits=None, temperature=self.temperature, guidance_scale=1.0, length_penalty=1.0)
                outputs = outputs + output
                scores = scores + score
                seq_scores.append(seq_score)

            for i, test in enumerate(test_data):
                test["predict"] = outputs[i]
                test["predict_score"] = scores[i]
                test["predict_seq_score"] = seq_scores[i]

            for i in range(len(test_data)):
                if 'dedup' in test_data[i]:
                    test_data[i].pop('dedup')  
    
            with open(self.path + '_' + str(self.train_theta) + '_' + str(self.gfn_theta) + '.json', 'w') as f:
                json.dump(test_data, f, indent=4)

            gao(self.path + '_' + str(self.train_theta) + '_' + str(self.gfn_theta) + '.json', self.info_file, self.result_data, self.group_title_path, self.category, str(round(int(epoch))))

    def on_epoch_end(self, args, state, control, **kwargs):
        self.perform_evaluation(args, state, control, 'epoch', **kwargs)

class MyTrainer(Trainer):
    def __init__(self, tokenizer, min_sentence_len, max_sentence_len, group_title_path, title_token_distribution, n_token_distribution, possible_output, gfn_theta, train_theta, n, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open(possible_output, 'r', encoding='utf-8') as file:
            temp = json.load(file)
        self.possible_output = list(temp.keys())
        
        self.tokenizer = tokenizer
        try:
            self.end_of_sentence_token_id = self.tokenizer.encode(
                "A sentence\n", add_special_tokens=False
            )[-1]
        except:
            self.end_of_sentence_token_id = self.tokenizer.convert_tokens_to_ids("\n")

        self.min_sentence_len = min_sentence_len
        self.max_sentence_len = max_sentence_len
        self.group_title_path = group_title_path
        self.title_token_distribution = title_token_distribution
        self.n_token_distribution = n_token_distribution
        self.n = n
        self.reward = FrozenModelSentenceGivenPrompt(
            sentence_token_id=self.end_of_sentence_token_id,
            min_len=self.min_sentence_len,
            group_title_path=self.group_title_path,
            tokenizer=self.tokenizer,
        )
        self.count = 0
        self.sft_time = 0
        self.gfn_time = 0
        self.gfn_theta = gfn_theta
        self.train_theta = train_theta
    
    def gfn_forward(self, prompt, model, pf_temperature=1.0, mode='train', action_seq=None):
        reward_fn = partial(
            self.reward.score,
            prompt_length=prompt['gfn_input'].shape[1],
            model=model,
            tokenizer=self.tokenizer,
        )
        (
            generated_text,
            log_pf,
            log_pterm,
            log_r,
            log_r_unpenalized,
            state_count
        ) = generate_and_return_termination_logprob(
            model,
            prompt['gfn_input'],
            reward_fn=reward_fn,
            tokenizer=self.tokenizer,
            termination_token_id=self.end_of_sentence_token_id,
            min_len=self.min_sentence_len,
            max_len=self.max_sentence_len,
            temperature=pf_temperature,
            skip_rewards=False,
            action_seq=action_seq,
            mode=mode,
            title_token_distribution=self.title_token_distribution,
            n_token_distribution=self.n_token_distribution,
            possible_output=self.possible_output,
            n=self.n
        )
        return generated_text, log_pf, log_pterm, log_r, log_r_unpenalized, state_count
    
    def compute_loss(self, model, inputs, return_outputs=False):
        sft_s_time = time.time()
        sft_input = {k:v for k,v in inputs.items() if k not in ['gfn_input','index']}
        outputs = model(**sft_input)
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        sft_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        sft_e_time = time.time()
        sft_time = sft_e_time - sft_s_time
        gfn_s_time = time.time()
        generated_text, log_pf, log_pterm, log_r, log_r_unpenalized, state_count = self.gfn_forward(
            inputs, model=model, action_seq=None, mode='train'
        )

        gfn_loss = modified_subtb_loss(
            log_pf=log_pf,
            log_r=log_r,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=inputs['gfn_input'].shape[1],
            subtb_lambda=1,
            state_count=state_count,
            n=self.n
        )
        gfn_e_time = time.time()
        gfn_time = gfn_e_time - gfn_s_time
        loss = self.train_theta*sft_loss + (1-self.train_theta)*gfn_loss
        return (loss, outputs) if return_outputs else loss

class CustomDataCollator(transformers.DataCollatorForSeq2Seq):
    def __call__(self, features):
        index = []
        gfn_input = []
        for i in range(len(features)):
            index.append(features[i]['index'])
            gfn_input.append(torch.tensor(features[i]['gfn_input']))
        max_length = max([tensor.shape[0] for tensor in gfn_input])
        n = len(gfn_input)
        pad_tensor = torch.full((n, max_length), fill_value=self.tokenizer.eos_token_id, dtype=torch.long)
        for i, tensor in enumerate(gfn_input):
            pad_tensor[i, :tensor.size(0)] = tensor

        features = [{k:v for k,v in f.items() if k not in ['gfn_input','index']} for f in features]
        batch = super().__call__(features)
        batch['gfn_input'] = pad_tensor
        batch['index'] = torch.tensor(index)
        return batch

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_file: str="",
    eval_file: str="",
    output_dir: str = "./lora-alpaca",
    sample: int = -1,
    seed: int = 0,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 7,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    local_rank: int = 0,
    deepspeed: str ="./deepspeed.json",
    category: str="",
    K: int = 0,
    version: str = "base",
    min_sentence_len: int = 1,
    max_sentence_len: int = 20,
    test_data_path: str = None,
    info_path: str = None,
    group_title_path: str = None,
    title_token_distribution: str = None,
    n_token_distribution: str = None,
    possible_output: str = None,
    temperature: float = 1.0,
    logits_file: str = None,
    gfn_theta: float = 0.9,
    train_theta: float = 0.999,
    n: int = 1,
):
    output_dir = os.path.join(output_dir, str(train_theta)+'_'+str(gfn_theta))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    f = open(info_path, 'r')
    items = f.readlines()
    item_names = [_.split('\t')[0].strip("\"").strip(" ").strip('\n').strip('\"') for _ in items]
    item_ids = [_ for _ in range(len(item_names))]
    id_title = dict()
    for i in range(len(item_names)):
        id_title[item_ids[i]] = item_names[i]

    category_dict = {"GoodReads": "books", "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors", "Steam": "games", 'Movielens':'movies'}
    print(category)
    origin_category = category
    category = category_dict[category]
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # uses.environ["WANDB_LOG_MODEL"] = wandb_log_model
    os.environ["WANDB_DISABLED"] = "true"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        # device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    train_data = D3Dataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category, K = K)
    val_data = D3Dataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, category=category, K = K)
        
    print("LOAD DATA FINISHED")

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    from datasets import Dataset as HFDataset
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()})
    trainer = MyTrainer(
        # deepspeed=deepspeed,
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        min_sentence_len=min_sentence_len,
        max_sentence_len=max_sentence_len,
        group_title_path=group_title_path,
        title_token_distribution=title_token_distribution,
        n_token_distribution=n_token_distribution,
        possible_output=possible_output,
        gfn_theta=gfn_theta,
        train_theta=train_theta,
        n=n,
        args=transformers.TrainingArguments(
            # deepspeed=deepspeed,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
            remove_unused_columns=False,
        ),
        data_collator=CustomDataCollator(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2), EvaluateCallback(test_data_path=test_data_path, category=origin_category, info_file=info_path, group_title_path=group_title_path, tokenizer=tokenizer, temperature=temperature, logits_file=logits_file, title_token_distribution=title_token_distribution, possible_output=possible_output, gfn_theta=gfn_theta, train_theta=train_theta, n=n)],
        # optimizers=(optimizer, lr_scheduler) 
    )
    model.config.use_cache = False
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
