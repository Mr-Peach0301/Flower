import os
os.environ['LD_LIBRARY_PATH'] = 'YOUR_CONDA_ENV/lib'
import sys
import json
from typing import List
import matplotlib.pyplot as plt
import random
from transformers import  LogitsProcessorList, TemperatureLogitsWarper
from LogitProcesser import CFEnhancedLogitsProcessor
import numpy as np 
import re
import scipy.stats
import math
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, GenerationConfig
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, TrainerCallback

def generate_prompt_test(category):
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Please recommend a {category} to the user. Directly output the title of the {category}.
### Response: 
"""
    return prompt

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
        with open(self.save_path + f'/title_{int(round(epoch))}.json', 'w') as f:
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
            plt.savefig(self.save_path + f'/{int(round(epoch))}.png', format='png', dpi=300, bbox_inches='tight')
 
            return np.sum(m * np.log(m / b)), np.sum(b * np.log(b / m)), JS_divergence(b,m)
            
        kl_mb, kl_bm, js_bm = kl_divergence_discrete(outputs, self.real_token_distri)

        with open(self.save_path + f'/divergence_discrete_{int(round(epoch))}.json', 'w') as f:
            json.dump({'kl_mb':kl_mb, 'kl_bm':kl_bm, 'js_bm':js_bm}, f, indent=4)

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_data_path: List[str] = [""],
    val_data_path: List[str] = [""],
    output_dir: str = "./lora-alpaca",
    sample: int = -1,
    seed: int = 0,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
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
    possible_output_path: str = None,
    real_token_distri: str = None,
    save_path: str = None,
    category: str = None,
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"seed: {seed}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    os.environ["WANDB_DISABLED"] = "true"
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = AutoModelForCausalLM.from_pretrained(
        base_model
    )
    # model.set_tau(tau)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(category, possible_output):
        full_prompt = generate_prompt(category, possible_output)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt(category, possible_output)
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt
    
    possible_output = []
    with open(possible_output_path, 'r', encoding='utf-8') as file:
        for line in file:
            possible_output.append(line.strip())
    
    category_dict = {'Movies': 'movies'}
    category = category_dict[category]

    with open(real_token_distri, "r") as input_file:
        titel_prob = json.load(input_file)
    total_sum = sum(titel_prob.values())
    title_prob = {key: value / total_sum for key, value in titel_prob.items()}


    train_data_list = []
    val_data_list = []

    for path in train_data_path:
        if path.endswith(".json"):
            train_data_list.append(load_dataset("json", data_files=path))
        else:
            train_data_list.append(load_dataset(path))

    for path in val_data_path:
        if path.endswith(".json"):
            val_data_list.append(load_dataset("json", data_files=path))
        else:
            val_data_list.append(load_dataset(path))

    for i in range(len(train_data_list)):
        train_data_list[i]["train"] = train_data_list[i]["train"].shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data_list[i]["train"].shuffle(seed=seed)
        train_data_list[i]["train"] = train_data_list[i]["train"].shuffle(seed=seed)
        train_data_list[i] = train_data_list[i].map(lambda x: generate_and_tokenize_prompt(category, title_prob))
    for i in range(len(val_data_list)):
        val_data_list[i] = val_data_list[i].map(lambda x: generate_and_tokenize_prompt(category, title_prob))
    train_data = concatenate_datasets([_["train"] for _ in train_data_list])
    val_data = concatenate_datasets([_["train"] for _ in val_data_list])

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    
 
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=8,
            optim="adamw_torch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10),EvaluateCallback(test_data_path=val_data_path[0], category=category, tokenizer=tokenizer, info_file=possible_output, real_token_distri=real_token_distri, save_path=save_path)]
    )
    model.config.use_cache = False

    trainer.train()

    model.save_pretrained(output_dir, safe_serialization=False)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(category, possible_output):
    output = random.choices(list(possible_output.keys()), weights=list(possible_output.values()), k=1)[0]
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Please recommend a {category} to the user. Directly output the title of the {category}.
### Response: 
{output}"""

if __name__ == "__main__":
    fire.Fire(train)
