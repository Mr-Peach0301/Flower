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
import scipy.stats
from transformers import  LogitsProcessorList, TemperatureLogitsWarper
from LogitProcesser import CFEnhancedLogitsProcessor
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from trl.core import LengthSampler
from accelerate import Accelerator

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

def get_tensors_from_dataframe(df_train, tokenizer, category):
    instructs = f"Please recommend a {category} to the user. Directly output the title of the {category}."
    instruction =  f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 
### Instruction:
{instructs}
### Response: 
"""
    tokens = tokenizer(instruction, return_tensors='pt')["input_ids"][0]
    target_item ='A fake item name'
    labels = tokenizer(target_item, return_tensors='pt')["input_ids"][0]
    input_ids = [tokens] * len(df_train)
    target_ids = [labels] * len(df_train)
    input_ids = pad_sequence(input_ids, padding_value=tokenizer.eos_token_id, batch_first=True)
    target_ids = pad_sequence(target_ids, padding_value=tokenizer.eos_token_id, batch_first=True)
    return input_ids, target_ids
 

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
    train_set: str = None,
    possible_output_path: str = None,
    num_samples_per_batch: int = 32,
    title_distri: str = None,
    category: str = 'movie',
    save_path: str = None,
):
    category_dict = {'Movies': 'movies'}
    category = category_dict[category]
    possible_output = []
    with open(possible_output_path, 'r', encoding='utf-8') as file:
        for line in file:
            possible_output.append(line.strip())
    with open(title_distri, "r") as input_file:
        t2i = json.load(input_file)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    def normalize_dict_values(d):
        values = list(d.values())
        min_val = min(values)
        max_val = max(values)

        if min_val == max_val:
            return {k: 0.5 for k in d}

        normalized_dict = {k: (v - min_val) / (max_val - min_val) for k, v in d.items()}

        return normalized_dict

    t2p = normalize_dict_values(t2i)

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
        ppo_epochs=ppo_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mini_batch_size=mini_batch_size,
        init_kl_coef=init_kl_coef,
        kl_penalty=kl_penalty,
        seed=seed
    )

    eos_string = tokenizer.decode(tokenizer.eos_token_id)

    with open(train_set, "r") as input_file:
        df_train = json.load(input_file)

    input_ids, target_ids = get_tensors_from_dataframe(df_train, tokenizer, category)
    train_dataset = TensorDataset(input_ids, target_ids)

    def collator(batch):
        input_ids, target_ids = zip(*batch)
        return {
            "input_ids": input_ids,
            "query": [s.replace(eos_string, "") for s in tokenizer.batch_decode(input_ids)],
            "target_ids": target_ids,
        }


    ppo_trainer = PPOTrainer(
        config=config,
        model=inference_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
    )

    generation_kwargs = {
        "max_new_tokens": 15,
        "top_k": 0,
        "top_p": 1,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    item_name = possible_output
    info = [f'''### Response: 
{_}''' for _ in item_name]
    
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

    output_min_length = 2
    output_max_length = 15
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
    
    for idex, batch in enumerate(tqdm(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
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
        for i in range(len(batch["response"])):
            if batch["response"][i][-1]==151643:
                batch["response"][i][-1]=198
        tag = []
        pro = []
        for response in response_tensors:
            decoded_text = tokenizer.decode(response, skip_special_tokens=True)
            print(decoded_text)
            if decoded_text in possible_output:
                tag.append(True)
                pro.append(t2p[decoded_text])
            else:
                tag.append(False)
                pro.append(-1)
        # Compute reward
        rewards = []
        # mean_rewards = 0
        for index, response in enumerate(batch["response"]):
            if tag[index] ==True:
                rewards.append(torch.tensor(float(pro[index])).cuda())
            else:
                rewards.append(torch.tensor(0.0001).cuda())
        stats = ppo_trainer.step(list(query_tensors), response_tensors, rewards)
        print(rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    p = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Please recommend a {category} to the user. Directly output the title of the {category}.
### Response: 
"""
    num_samples = len(possible_output)*20
    instructions = [p for _ in range(num_samples)]

    inference_model.eval()
    result = {}
    with torch.inference_mode():
        def batch(list, batch_size=num_samples_per_batch):
            chunk_size = (len(list) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield list[batch_size * i: batch_size * (i + 1)]
        for i, b in tqdm(enumerate(batch(instructions))):
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
            instructions = b
            inputs = tokenizer(instructions, return_tensors="pt", padding=True, truncation=True).to('cuda')
            generated_outputs = inference_model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True, 
                top_k=0,
                top_p=1.0,
                temperature=1.0,
                num_return_sequences=1,
                max_new_tokens=20,
                logits_processor=logits_processor,
            )
            s = generated_outputs.sequences
            output = tokenizer.batch_decode(s, skip_special_tokens=True)
            output = [_.split('Response: \n')[-1] for _ in output]
            for generated_output in output:
                if generated_output in result:
                    result[generated_output] += 1
                else:
                        result[generated_output] = 1
        with open(save_path + f'/title.json', 'w') as f:
            json.dump(result, f, indent=4)
        
        float_info = np.finfo(np.float64)

        def JS_divergence(p,q):
            M=(p+q)/2
            return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)

        def kl_divergence_discrete(before, ml):
            with open(ml, "r") as input_file:
                data = json.load(input_file)
            keys = set(before.keys()).union(data.keys())
            b_counts = [before.get(key,float_info.eps) for key in keys]
            c_counts = [data.get(key, float_info.eps) for key in keys]
            b_total = sum(b_counts)
            c_total = sum(c_counts)
            b_dist = [count / b_total for count in b_counts]
            c_dist = [count / c_total for count in c_counts]
            combined = sorted(zip(c_dist, b_dist), key=lambda x: x[0], reverse=True)
            sorted_a, sorted_b = zip(*combined)
            c_d = list(sorted_a)
            b_d = list(sorted_b)
            b = np.asarray(list(b_dist), dtype=np.float64)
            m = np.asarray(list(c_dist), dtype=np.float64)
            plt.figure()
            plt.bar(list(keys), b_d, label='Result Distribution', alpha=0.5)
            plt.bar(list(keys), c_d, label='Target Distribution', alpha=0.5)
            plt.legend()
            plt.title('Comparison of Two Distributions')
            plt.xlabel('Categories')
            plt.ylabel('Values')
            plt.savefig(save_path + f'/fig.png', format='png', dpi=300, bbox_inches='tight')
 
            return np.sum(m * np.log(m / b)), np.sum(b * np.log(b / m)), JS_divergence(b,m)
            
        kl_mb, kl_bm, js_bm = kl_divergence_discrete(result, title_distri)

        with open(save_path + f'/divergence_discrete.json', 'w') as f:
            json.dump({'kl_mb':kl_mb, 'kl_bm':kl_bm, 'js_bm':js_bm}, f, indent=4)


if __name__ == "__main__":
    fire.Fire(train)