import torch
import heapq
import pickle
import gzip
import editdistance
import spacy
import numpy as np
import json
from torch.nn.utils.rnn import pad_sequence
import math

def get_hash(x):
    x = [str(int(_)) for _ in x]
    return '-'.join(x)
def get_state_chain(x):
    x = [str(_) for _ in x]
    return '@'.join(x)

@torch.no_grad()
def score_fast(
    model,
    encoded_input,
    termination_token_id,
    min_len,
    skip_first,
    prompt_cache=None,
    group_title_path=None,
    sentence_token_id=None,
    tokenizer=None,
    method='online',
    title_token_distribution=None,
    possible_output=None,
    n=1
):
    if prompt_cache is None:
        logits = model(encoded_input).logits
    else:
        batched_prompt_cache = tuple(
            tuple(
                [
                    prompt_cache[1][i][j].repeat(encoded_input.shape[0], 1, 1, 1)
                    for j in range(len(prompt_cache[1][i]))
                ]
            )
            for i in range(len(prompt_cache[1]))
        )
        logits = model(encoded_input, past_key_values=batched_prompt_cache).logits
    logits = logits[:, skip_first - 1 :]
    logprob = logits.log_softmax(-1)
    token_ids = encoded_input[:, skip_first:].unsqueeze(-1)
    
    with open(group_title_path, "r") as input_file:
        group_title = json.load(input_file)

    tag = []
    for index, input_ids in enumerate(encoded_input[:, skip_first:]):
        if termination_token_id in input_ids.tolist():
            idx = input_ids.tolist().index(termination_token_id)
            decoded_text = tokenizer.decode(input_ids[:idx], skip_special_tokens=True)
        else:
            decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        for j in group_title.keys():
            if decoded_text in group_title[j]:
                tag.append(j)
                break
    
    with open(title_token_distribution, "r") as input_file:
        token_distri = json.load(input_file)

    complete_title = []
    term_prob = []
    state_count = []
    
    logr = torch.full((logprob.shape[0], math.ceil((token_ids.shape[1])/n)+1), -15.0*n, device='cuda')
    
    for i in range(logprob.shape[0]):
        mask = token_ids[i] == termination_token_id
        indices = torch.nonzero(mask, as_tuple=True)[0]
        if len(indices) > 0:
            length = indices[0].item()
        else:
            length = token_ids.shape[1]
        state_num = math.ceil(length/n)
        state_count.append(state_num)
        state_list = [get_hash(token_ids[i][j*n:min(length,(j+1)*n)]) for j in range(state_num)]
        for j in range(state_num):
            current_state = state_list[j]
            if j==0:
                if current_state in token_distri['first']:
                    logr[i, j] = token_distri['first'][current_state]
                else:
                    logr[i, j:] = -15*n
                    complete_title.append(-1)
                    term_prob.append(-15)
                    break 
            elif j==1:
                if state_list[0] in token_distri and current_state in token_distri[state_list[0]]:
                    logr[i, j] = token_distri[state_list[0]][current_state]
                else:
                    logr[i, j:] = -15*n
                    complete_title.append(-1)
                    term_prob.append(-15)
                    break
            else:
                if get_state_chain(state_list[:j]) in token_distri and current_state in token_distri[get_state_chain(state_list[:j])]:
                    logr[i, j] = token_distri[get_state_chain(state_list[:j])][current_state]
                else:
                    logr[i, j:] = -15*n
                    complete_title.append(-1)
                    term_prob.append(-15)
                    break
            if j==state_num-1:
                if tokenizer.decode(token_ids[i, :length, 0].tolist()) in possible_output:
                    complete_title.append(state_num)
                    term_prob.append(token_distri[get_state_chain(state_list)][str(termination_token_id)])
                else:
                    complete_title.append(-1)
                    term_prob.append(-15)

    logPF = logr[:, :-1]
    logP = logPF.cumsum(dim=-1)  # logP(generated[:i+1] | prompt)
    logterm = torch.full((logr.shape[0], logr.shape[1]), -15.0, device='cuda')

    for i in range(token_ids.shape[0]):
        if complete_title[i] !=-1:
            logterm[i, complete_title[i]] = term_prob[i]
    reward = logterm # logP(generated[i+1]=term | prompt + generated[:i])
    reward[:, 1:] += logP  # logP(generated[:i] + term | prompt)
    mask = torch.ones_like(reward, dtype=torch.bool)

    for i, idx in enumerate(state_count):
        if idx+1 < reward.size(1):
            mask[i, idx+1:] = False 

    reward[~mask] = 0.0
    reward_unpenalized = reward.clone()
    return reward, reward_unpenalized, encoded_input, state_count


class FrozenModelSentenceGivenPrompt:
    def __init__(
        self,
        sentence_token_id,
        temperature=1.0,
        min_len=1,
        group_title_path = None,
        tokenizer = None
    ):

        self.temperature = temperature
        self.sentence_token_id = sentence_token_id
        self.min_len = min_len
        self.group_title_path = group_title_path
        self.tokenizer = tokenizer

    def score(self, input_batch, prompt_length, model, tokenizer, method, title_token_distribution, possible_output, n):
        training = model.training
        model.eval()
        reward, reward_unpenalized, encoded_input, state_count = score_fast(
            model=model,
            encoded_input=input_batch,
            termination_token_id=self.sentence_token_id,
            skip_first=prompt_length,
            min_len=self.min_len,
            group_title_path=self.group_title_path,
            sentence_token_id=self.sentence_token_id,
            tokenizer=self.tokenizer,
            method=method,
            title_token_distribution=title_token_distribution,
            possible_output=possible_output,
            n=n
        )
        if training:
            model.train()
        return reward, reward_unpenalized, encoded_input, state_count


def generate_and_return_termination_logprob(
    model,
    encoded_prompt,
    termination_token_id,
    reward_fn,
    tokenizer,
    max_len=10,
    min_len=0,
    temperature=1,
    top_k=999999,
    top_p=1.0,
    action_seq=None,
    skip_rewards=False,
    mode='train',
    title_token_distribution=None,
    n_token_distribution=None,
    possible_output=None,
    n=1
):
    with open(title_token_distribution, "r") as input_file:
        token_distri = json.load(input_file)
    active_seqs = torch.ones(encoded_prompt.size(0)).bool().to(encoded_prompt.device)
    state = encoded_prompt.clone()
    log_pf = []
    log_pterm = []
    logging_token = []
    token_ids = state 
    past_key_values = None
    for i in range(max_len + 1):
        output = model(input_ids=token_ids, past_key_values=past_key_values)
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]
        if action_seq is None:
            with torch.no_grad():
                modified_logits = logits.clone().detach()
                if i==0:
                    all_columns = torch.arange(modified_logits.shape[1])
                    update_columns = ~torch.isin(all_columns, torch.tensor([int(_) for _ in token_distri['first'].keys()]))
                    modified_logits[:, update_columns] = -torch.inf
                elif i==1:
                    for j in range(encoded_prompt.size(0)):
                        all_columns = torch.arange(modified_logits.shape[1])
                        update_columns = ~torch.isin(all_columns, torch.tensor([int(_) for _ in token_distri[str(state[j][int(encoded_prompt[0].shape[0])].item())].keys()]))
                        modified_logits[j, update_columns] = -torch.inf
                else:
                    for j in range(encoded_prompt.size(0)):
                        if state[j][int(encoded_prompt[0].shape[0])+i-1].item() == termination_token_id:
                            continue
                        all_columns = torch.arange(modified_logits.shape[1])
                        update_columns = ~torch.isin(all_columns, torch.tensor([int(_) for _ in token_distri[get_hash(state[j][int(encoded_prompt[0].shape[0]):int(encoded_prompt[0].shape[0])+i])].keys()]))
                        modified_logits[j, update_columns] = -torch.inf
                prob = modified_logits.softmax(dim=-1)
                token_ids = torch.multinomial(prob, num_samples=1)
        else:
            if i >= action_seq.size(-1):
                token_ids = (
                    torch.ones_like(action_seq[:, 0]) * termination_token_id
                ).unsqueeze(-1)
            else:
                token_ids = action_seq[:, i].unsqueeze(-1)
        token_ids = torch.where(
            active_seqs.unsqueeze(-1),
            token_ids,
            termination_token_id,
        )
        logging_token.append(token_ids)
        logprob = logits.log_softmax(dim=-1)
        log_pterm.append(
            torch.where(
                active_seqs,
                logprob[:, termination_token_id],
                0,
            )
        )
        active_seqs = active_seqs * (token_ids != termination_token_id).squeeze(-1)
        log_pf.append(
            torch.where(
                active_seqs,
                logprob.gather(-1, token_ids).squeeze(-1),
                0,
            )
        )
        state = torch.cat([state, token_ids], dim=-1)
        if torch.all(~active_seqs):
            break
    log_pf = torch.stack(log_pf, dim=1)
    log_pterm = torch.stack(log_pterm, dim=1)
    logging_token = torch.stack(logging_token, dim=1)

    n_log_pf = []
    n_log_pterm = []
    for i in range(logging_token.shape[0]):
        mask = logging_token[i] == termination_token_id
        indices = torch.nonzero(mask, as_tuple=True)[0]
        if len(indices) > 0:
            length = indices[0].item()
            flag = 0
        else:
            length = log_pf.shape[1]-1
            flag = 1
        state_num = math.ceil(length/n)

        indices = torch.arange(start=0, end=length, step=n)
        l_pt = log_pterm[i][indices]
        l_pt = torch.cat((l_pt, log_pterm[i, length].unsqueeze(0)))
        
        l_pf = []
        for j in range(state_num):
            l_pf.append(log_pf[i, j*n:min(length,(j+1)*n)].sum())
        l_pf = torch.stack(l_pf)
        if flag==0:
            l_pf = torch.cat((l_pf, torch.tensor([0.0], dtype=l_pf.dtype, device=l_pf.device)))
        else:
            l_pf = torch.cat((l_pf, log_pf[i, length].unsqueeze(0)))

        n_log_pf.append(l_pf)
        n_log_pterm.append(l_pt)
    n_log_pterm = pad_sequence(n_log_pterm, batch_first=True, padding_value=0)
    n_log_pf = pad_sequence(n_log_pf, batch_first=True, padding_value=0)

    if action_seq is None:
        method='online'
    else:
        method='offline'
    if skip_rewards:
        log_r, log_r_unpenalized = None, None
    else:
        log_r, log_r_unpenalized, encoded_input, state_count = reward_fn(state[:, :-1], method=method, title_token_distribution=n_token_distribution, possible_output=possible_output, n=n)
        encoded_input = torch.cat((encoded_input, state[:, -1].unsqueeze(-1)), dim=1)
    return encoded_input, n_log_pf, n_log_pterm, log_r, log_r_unpenalized, state_count


def modified_subtb_loss(
    log_pf,
    log_r,
    log_pterm,
    generated_text,
    termination_token_id,
    prompt_len,
    subtb_lambda=1.0,
    state_count=None,
    n=1
):
    assert (
        log_pf.shape[1]
        == log_r.shape[1]
        == log_pterm.shape[1]
    )
    assert (
        log_pf.shape[1] > 1
    ) 

    delta = (
        log_r[:, :-1]
        + log_pf[:, :-1]
        + log_pterm[:, 1:]
        - log_r[:, 1:]
        - log_pterm[:, :-1]
    )
    delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], 1).cumsum(1)

    mask = torch.zeros((generated_text.shape[0], math.ceil((generated_text.shape[1]-prompt_len-1)/n)), dtype=torch.bool, device=generated_text.device)
    
    for i, idx in enumerate(state_count):
        if idx < mask.size(1): 
            mask[i, idx:] = True

    batch_loss = 0.0
    total_lambda = 0.0

    generated_len = math.ceil((generated_text.shape[1] - prompt_len -1)/n)+1
    
    for subtraj_len in range(1, generated_len):
        subtb_term = (
            delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]
        ) ** 2
        subtb_term[mask[:, subtraj_len - 1 :]] = 0
        batch_loss += subtb_lambda ** (subtraj_len - 1) * subtb_term.sum()
        total_lambda += (
            subtb_lambda ** (subtraj_len - 1) * (~mask[:, subtraj_len - 1 :]).sum()
        )
    batch_loss /= total_lambda
    return batch_loss