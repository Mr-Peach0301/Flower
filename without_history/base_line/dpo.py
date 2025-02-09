import os
from typing import List
import numpy as np 
import fire
import torch
import json
import random
import scipy.stats
import matplotlib.pyplot as plt
from LogitProcesser import CFEnhancedLogitsProcessor
from datasets import load_dataset
import torch.nn as nn
import math
import pandas as pd
from tqdm import tqdm
from functools import partial
import fire
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, LogitsProcessorList, TemperatureLogitsWarper
from accelerate import Accelerator
from trl import (
    DPOConfig,
    DPOTrainer
)

def generate_prompt_test(category):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Please recommend a {category} to the user. Directly output the title of the {category}.
### Response: 
"""

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

class EvaluateCallback(TrainerCallback):
    def __init__(self, test_data_path, category, tokenizer, info_file, real_token_distri, save_path):
        self.test_data_path = test_data_path
        self.category = category
        self.tokenizer = tokenizer
        self.batch_size = 32
        self.info_file = info_file
        self.real_token_distri = real_token_distri
        self.save_path = save_path

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()

        item_name = self.info_file
        info = [f'''### Response: 
{_}''' for _ in item_name]
    
        prefixID = [self.tokenizer(_).input_ids for _ in info]

        hash_dict = dict()
        sasrec_dict = dict()
        for index, ID in enumerate(prefixID):
            ID.append(self.tokenizer.eos_token_id)
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

        def evaluate(
            instructions,
            inputs=None,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            num_beams=1,
            max_new_tokens=128,
            category=None,
            tokenizer=None,
            **kwargs,
        ):
            prompt = [generate_prompt_test(category) for instruction, input in zip(instructions, inputs)]
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to('cuda')
            with torch.no_grad():
                ccc = CFEnhancedLogitsProcessor(
                    guidance_scale=1,
                    cf_logits=None,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    cf_dict=sasrec_dict,
                    unconditional_ids=None,
                    model=model,
                    tokenizer=tokenizer,
                    num_beams=num_beams
                )
                logits_processor = LogitsProcessorList([TemperatureLogitsWarper(temperature=1.0), ccc])
            
                generation_output = model.generate(
                    **inputs,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=True, 
                    top_k=0,
                    top_p=1.0,
                    temperature=1.0,
                    num_return_sequences=1,
                    max_length=150,
                    logits_processor=logits_processor,
                )
                                
    
            s = generation_output.sequences
            output = tokenizer.batch_decode(s, skip_special_tokens=True)
            output = [_.split('Response: \n')[-1] for _ in output]
            return output

        epoch = state.epoch
        outputs = []
        from tqdm import tqdm
        import json
        with open(self.test_data_path, 'r') as f:
            test_data = json.load(f)
        test_data = test_data
        instructions = [_['instruction'] for _ in test_data]
        inputs = [_['input'] for _ in test_data]
        def batch(list, batch_size=self.batch_size):
            chunk_size = (len(list) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield list[batch_size * i: batch_size * (i + 1)]
        outputs = {}
        for i, batch in tqdm(enumerate(zip(batch(instructions), batch(inputs)))):
            instructions, inputs = batch
            output = evaluate(instructions, inputs, category=self.category, tokenizer=self.tokenizer)
            print(output)
            for i in output:
                if i in outputs:
                    outputs[i] += 1
                else:
                    outputs[i] = 1
        with open(self.save_path + f'/title_{epoch}.json', 'w') as f:
            json.dump(outputs, f, indent=4)
        
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
            plt.savefig(self.save_path + f'/{epoch}.png', format='png', dpi=300, bbox_inches='tight')
 
            return np.sum(m * np.log(m / b)), np.sum(b * np.log(b / m)), JS_divergence(b,m)
            
        kl_mb, kl_bm, js_bm = kl_divergence_discrete(outputs, self.real_token_distri)

        with open(self.save_path + f'/divergence_discrete_{epoch}.json', 'w') as f:
            json.dump({'kl_mb':kl_mb, 'kl_bm':kl_bm, 'js_bm':js_bm}, f, indent=4)


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_file: str="",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 1,
    micro_batch_size: int = 1,
    num_epochs: int = 1,
    learning_rate: float = 1e-6,
    cutoff_len: int = 512,
    category: str="",
    test_data_path: str = None,
    beta: float = 0.3,
    seed: int = 66,
    possible_output_path: str = None,
    real_token_distri: str = None,
    save_path: str = None,
):
    possible_output = []
    with open(possible_output_path, 'r', encoding='utf-8') as file:
        for line in file:
            possible_output.append(line.strip())
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    category_dict = {'Movies':'movies'}

    origin_category = category
    category = category_dict[category]

    gradient_accumulation_steps = batch_size // micro_batch_size
    
    device_index = Accelerator().process_index
    device_map = {"": device_index}

    os.environ["WANDB_DISABLED"] = "true"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

    train_dataset = load_dataset("json", data_files=train_file)
    train_data = train_dataset["train"].shuffle(seed=seed)
    print("LOAD DATA FINISHED")
    
    training_args = DPOConfig(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=20,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=1,
        optim="adamw_torch",
        save_strategy="epoch",
        output_dir=output_dir,
        save_total_limit=1,
        report_to=None,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        train_dataset=train_data,
        beta=beta,
        tokenizer=tokenizer,
        max_prompt_length=cutoff_len,
        max_length=cutoff_len,
        args=training_args,
        callbacks = [EvaluateCallback(test_data_path=test_data_path, category=origin_category, info_file=possible_output, tokenizer=tokenizer, real_token_distri=real_token_distri, save_path=save_path)]
    )
    
    trainer.train()

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
