import random
from functools import partial
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import time
from collections import defaultdict
import json
import os
import scipy.stats
from transformers import GenerationConfig
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import pad_sequence
from utils import (
    generate_and_return_termination_logprob,
    modified_subtb_loss,
)


class NextSentenceGFNTask(LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        reward,
        lr,
        accumulate_grad_batches,
        category,
        subtb_lambda,
        min_sentence_len,
        max_sentence_len,
        valid_path = None,
        test_data_path = None,
        title_logp_path = None,
        possible_output=None,
        batch_size=32,
        use_4bit=False,
        real_token_distri=None,
        discount_factor=0.9,
        token_distri=None,
        threshold=None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tokenizer"])

        self.model = model
        self.tokenizer = tokenizer
        self.reward = reward
        self.valid_path = valid_path
        if not os.path.exists(self.valid_path):
            os.makedirs(self.valid_path)
        self.test_data_path = test_data_path
        self.accumulate_grad_batches = accumulate_grad_batches
        self.title_logp_path = title_logp_path,
        self.possible_output = possible_output
        self.batch_size = batch_size
        self.real_token_distri = real_token_distri
        self.category = category
        self.discount_factor = discount_factor
        self.token_distri=token_distri
        self.log_train_time=0
        self.batch_num=0
        self.train_loss=0
        self.use_buffer_prob=1.0
        self.threshold=threshold

        self.get_lr_at_step = lambda step: min(step / 20 * lr, lr)
        try:
            self.end_of_sentence_token_id = tokenizer.encode(
                "A sentence\n", add_special_tokens=False
            )[-1]
        except:
            self.end_of_sentence_token_id = tokenizer.convert_tokens_to_ids("\n")

    def forward(self, prompt, pf_temperature=1.0, mode='train', action_seq=None):
        prompt = prompt[:,0,:]
        reward_fn = partial(
            self.reward.score,
            prompt_length=prompt.shape[1],
            model=self.model,
            tokenizer=self.tokenizer,
            discount_factor=self.discount_factor
        )
        (
            generated_text,
            log_pf,
            log_pterm,
            log_r,
            log_r_unpenalized,
        ) = generate_and_return_termination_logprob(
            self.model,
            prompt,
            reward_fn=reward_fn,
            tokenizer=self.tokenizer,
            termination_token_id=self.end_of_sentence_token_id,
            min_len=self.hparams.min_sentence_len,
            max_len=self.hparams.max_sentence_len,
            temperature=pf_temperature,
            skip_rewards=False,
            action_seq=action_seq,
            mode=mode,
            real_token_distri=self.token_distri
        )
        return generated_text, log_pf, log_pterm, log_r, log_r_unpenalized

    def training_step(self, prompt, batch_idx):
        flag = 'offline'
        if (
            random.random() < self.use_buffer_prob
        ):
            
            weights = [len(self.tokenizer.encode(item)) for item in self.possible_output]
            su = sum(weights)
            weights = [w/su for w in weights]
            labels = random.choices(self.possible_output, weights=weights, k=prompt.shape[0])

            labels = [torch.tensor(self.tokenizer.encode(m)) for m in labels]    
            labels = pad_sequence(labels, batch_first=True, padding_value=self.end_of_sentence_token_id)

            generated_text, log_pf, log_pterm, log_r, log_r_unpenalized = self.forward(
                prompt, action_seq=labels.to('cuda'), mode='train'
            )
        else:
            flag = 'online'
            generated_text, log_pf, log_pterm, log_r, log_r_unpenalized = self.forward(
                prompt, mode='train'
            )
            print('!!!')
            
        gfn_loss = modified_subtb_loss(
            log_pf=log_pf,
            log_r=log_r,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=prompt.shape[2],
            subtb_lambda=self.hparams.subtb_lambda,
        )
        self.log(
            "train/gfn_loss",
            gfn_loss,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
        )
        if flag=='offline' and self.current_epoch!=0:
            self.batch_num = self.batch_num + 1
            self.train_loss += gfn_loss
        return gfn_loss

    def validation_step(self, prompt, batch_idx):
        generated_text, log_pf, log_pterm, log_r, log_r_unpenalized = self.forward(
            prompt, mode='train'
        )

        loss = modified_subtb_loss(
            log_pf=log_pf,
            log_r=log_r,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=prompt.shape[2],
            subtb_lambda=self.hparams.subtb_lambda,
        )
        self.log(
            "val/loss",
            loss,
            sync_dist=True,
            prog_bar=True,
        )
    
    def on_train_batch_start(self, prompt, batch_idx):
        lr = self.get_lr_at_step(self.global_step)
        for pg in self.optimizers().param_groups:
            pg["lr"] = lr

    
    def on_validation_epoch_end(self):
        with torch.no_grad():
            if self.current_epoch in range(1):
                return
            outputs = {}
            with open(self.test_data_path, 'r') as f:
                test_data = json.load(f)
                test_data = test_data
                instructions = [_['instruction'] for _ in test_data]
                inputs = [_['input'] for _ in test_data]
                def batch(list, batch_size=16):
                    chunk_size = (len(list) - 1) // batch_size + 1
                    for i in range(chunk_size):
                        yield list[batch_size * i: batch_size * (i + 1)]
                for i, batch in tqdm(enumerate(zip(batch(instructions), batch(inputs)))):
                    instr, inp = batch
                    output = self.evaluate(instr, inp)
                    for op in output:
                        if op not in outputs:
                            outputs[op] = 1
                        else:
                            outputs[op] += 1

            op = {}
            for d,v in outputs.items():
                if '\n' in d:
                    index = d.index('\n')
                    d = d[:index]
                if d in op:
                    op[d] += v
                else:
                    op[d] = v
            outputs = op

            with open(self.valid_path+str(self.current_epoch)+'title.json', 'w') as f:
                json.dump(outputs, f, indent=4)
            
            float_info = np.finfo(np.float64)

            def JS_divergence(p,q):
                M=(p+q)/2
                return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)

            def kl_divergence_discrete(before, ml):
                with open(ml, "r") as input_file:
                    data = json.load(input_file)
                temp = {}
                for k,v in before.items():
                    if k in data:
                        temp[k] = v
                    else:
                        if 'Not_in_dataset' in temp:
                            temp['Not_in_dataset'] += 1
                        else:
                            temp['Not_in_dataset'] = 1
                before = temp
                data['Not_in_dataset'] = float_info.eps
                keys = set(before.keys()).union(data.keys())
                b_counts = [before.get(key,float_info.eps) for key in keys]
                c_counts = [data.get(key, float_info.eps) for key in keys]
                b_total = sum(b_counts)
                c_total = sum(c_counts)
                b_dist = [count / b_total for count in b_counts]
                c_dist = [count / c_total for count in c_counts]
                b = np.asarray(list(b_dist), dtype=np.float64)
                m = np.asarray(list(c_dist), dtype=np.float64)
 
                return np.sum(m * np.log(m / b)), np.sum(b * np.log(b / m)), JS_divergence(b,m)
            
            kl_mb, kl_bm, js_bm = kl_divergence_discrete(outputs, self.real_token_distri)

            if self.batch_num != 0:
                with open(self.valid_path+str(self.current_epoch)+'divergence_discrete.json', 'w') as f:
                    json.dump({'kl_mb':kl_mb, 'kl_bm':kl_bm, 'js_bm':js_bm, 'sec/batch':self.log_train_time/self.batch_num, 'sec/epoch':self.log_train_time}, f, indent=4)
                if self.train_loss/self.batch_num < self.threshold and self.use_buffer_prob > 0.15:
                    self.use_buffer_prob = self.use_buffer_prob - 0.1
                print('@@@@@@@@')
                print(self.train_loss/self.batch_num)
            self.log_train_time = 0
            self.batch_num = 0
            self.train_loss = 0
            
            self.log(
                "js_bm",
                js_bm,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.log(
                "use_buffer_prob",
                self.use_buffer_prob,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

    def configure_optimizers(self):
        if self.hparams.use_4bit:
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW8bit(self.model.parameters(), lr=self.hparams.lr)
        else:
            return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        

    def evaluate(
        self,
        instructions,
        inputs=None,
        **kwargs,
    ):
         
        prompt = [generate_prompt(instruction, self.category, input) for instruction, input in zip(instructions, inputs)]
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to('cuda')
        generated_text, log_pf, log_pterm, log_r, log_r_unpenalized = self.forward(
            inputs["input_ids"].unsqueeze(1), mode='eval'
        )
        output = []
        for s in generated_text:
            output.append(self.tokenizer.decode(s[int(inputs["input_ids"][0].shape[0]):]))
        return output
    
def generate_prompt(instruction, category, input=None):
    if input:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Please recommend a {category} to the user. Directly output the title of the {category}.
### Response:
"""
    
def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)