import torch
import heapq
import pickle
import gzip
import editdistance
import spacy
import numpy as np
import json

def get_hash(x):
    x = [str(int(_)) for _ in x]
    return '-'.join(x)

@torch.no_grad()
def score_fast(
    model,
    encoded_input,
    termination_token_id,
    min_len,
    skip_first,
    prompt_cache=None,
    title_logp_path=None,
    sentence_token_id=None,
    tokenizer=None,
    real_token_distri=None,
    discount_factor=0.8,
    method='online'
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
    
    possible_output = []
    with open(title_logp_path, 'r', encoding='utf-8') as file:
        for line in file:
            possible_output.append(line.strip())

    tag = []

    for index, input_ids in enumerate(encoded_input[:, skip_first:]):
        if termination_token_id in input_ids.tolist():
            idx = input_ids.tolist().index(termination_token_id)
            decoded_text = tokenizer.decode(input_ids[:idx], skip_special_tokens=True)
        else:
            decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        if decoded_text in possible_output:
            tag.append(True)
        else:
            tag.append(False)


    with open(real_token_distri, "r") as input_file:
        token_distri = json.load(input_file)

    complete_title = []
    term_prob = []

    logr = logprob.clone()
    for i in range(logprob.shape[0]):
        for j in range(token_ids.shape[1]):
            token = token_ids[i, j, 0].item()
            if j==0:
                if str(token) in token_distri['first']:
                    logr[i, j, token] = token_distri['first'][str(token)]
                else:
                    logr[i, j:, :] = -15
                    complete_title.append(-1)
                    term_prob.append(-15)
                    break
            elif j==1:
                if str(token_ids[i, :j, 0].item()) in token_distri and str(token) in token_distri[str(token_ids[i, :j, 0].item())]:
                    logr[i, j, token] = token_distri[str(token_ids[i, :j, 0].item())][str(token)]
                else:
                    logr[i, j:, :] = -15
                    complete_title.append(-1)
                    term_prob.append(-15)
                    break
            else:
                if get_hash(token_ids[i, :j, 0].tolist()) in token_distri and str(token) in token_distri[get_hash(token_ids[i, :j, 0].tolist())]:
                    logr[i, j, token] = token_distri[get_hash(token_ids[i, :j, 0].tolist())][str(token)]
                else:
                    logr[i, j:, :] = -15
                    if token_ids[i, j-1, 0].item()==termination_token_id and tokenizer.decode(token_ids[i, :j-1, 0].tolist()) in possible_output:
                        complete_title.append(j-1)
                        term_prob.append(token_distri[get_hash(token_ids[i, :j-1, 0].tolist())][str(termination_token_id)])
                    else:
                        complete_title.append(-1)
                        term_prob.append(-15)
                    break
            if j==token_ids.shape[1]-1 and token!=termination_token_id:
                if tokenizer.decode(token_ids[i, :, 0].tolist()) in possible_output:
                    complete_title.append(token_ids.shape[1])
                    term_prob.append(token_distri[get_hash(token_ids[i, :, 0].tolist())][str(termination_token_id)])
                else:
                    complete_title.append(-1)
                    term_prob.append(-15)
            elif j==token_ids.shape[1]-1 and token==termination_token_id:
                complete_title.append(token_ids.shape[1]-1)
                term_prob.append(token_distri[get_hash(token_ids[i, :j, 0].tolist())][str(termination_token_id)])
    logPF = logr[:, :-1].gather(-1, token_ids).squeeze(-1)
    logP = logPF.cumsum(dim=-1)

    for i in range(token_ids.shape[0]):
        if complete_title[i] !=-1:
            logprob[i, :complete_title[i],termination_token_id] = -15
            logprob[i, complete_title[i],termination_token_id] = term_prob[i]
        else:
            logprob[i, :,termination_token_id] = -15
    reward = logprob[
        :, :, termination_token_id
    ]
    reward[:, 1:] += logP
    non_term_mask = (encoded_input != termination_token_id)[:, skip_first:]
    non_term_mask = torch.cat(
        (
            non_term_mask.new_ones(non_term_mask.shape[0], 1),
            non_term_mask,
        ),
        dim=-1,
    )
    reward[~non_term_mask] = 0.0
    reward_unpenalized = reward.clone()
    return reward, reward_unpenalized, encoded_input


class FrozenModelSentenceGivenPrompt:
    def __init__(
        self,
        sentence_token_id,
        min_len=1,
        title_logp_path = None,
        tokenizer = None,
        real_token_distri = None
    ):
        self.sentence_token_id = sentence_token_id
        self.min_len = min_len
        self.title_logp_path = title_logp_path
        self.tokenizer = tokenizer
        self.real_token_distri = real_token_distri

    def score(self, input_batch, prompt_length, model, tokenizer, discount_factor, method):
        training = model.training
        model.eval()
        reward, reward_unpenalized, encoded_input = score_fast(
            model=model,
            encoded_input=input_batch,
            termination_token_id=self.sentence_token_id,
            skip_first=prompt_length,
            min_len=self.min_len,
            title_logp_path=self.title_logp_path,
            sentence_token_id=self.sentence_token_id,
            tokenizer=self.tokenizer,
            real_token_distri=self.real_token_distri,
            discount_factor=discount_factor,
            method=method
        )
        if training:
            model.train()
        return reward, reward_unpenalized, encoded_input


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
    real_token_distri=None
):
    with open(real_token_distri, "r") as input_file:
        token_distri = json.load(input_file)
    active_seqs = torch.ones(encoded_prompt.size(0)).bool().to(encoded_prompt.device)
    state = encoded_prompt.clone()
    log_pf = []
    log_pterm = []
    token_ids = state
    past_key_values = None
    for i in range(max_len + 1):
        output = model(input_ids=token_ids, past_key_values=past_key_values)
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]
        if action_seq is None:
            with torch.no_grad():
                prob = logits.softmax(dim=-1)

                modified_logits = logits.clone().detach()
                
                if mode=='eval':
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
                
                prob = (modified_logits / temperature).softmax(dim=-1)

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
    print('--------')
    for s in state:
        print(tokenizer.decode(s[int(encoded_prompt[0].shape[0]):]))
    log_pf = torch.stack(log_pf, dim=1)
    log_pterm = torch.stack(log_pterm, dim=1)

    if action_seq is None:
        method='online'
    else:
        method='offline'
    if skip_rewards:
        log_r, log_r_unpenalized = None, None
    else:
        log_r, log_r_unpenalized, encoded_input = reward_fn(state[:, :-1], method=method)
        encoded_input = torch.cat((encoded_input, state[:, -1].unsqueeze(-1)), dim=1)
    return encoded_input, log_pf, log_pterm, log_r, log_r_unpenalized


def modified_subtb_loss(
    log_pf,
    log_r,
    log_pterm,
    generated_text,
    termination_token_id,
    prompt_len,
    subtb_lambda=1.0,
):
    assert (
        log_pf.shape[1]
        == log_r.shape[1]
        == log_pterm.shape[1]
        == generated_text.shape[1] - prompt_len
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

    mask = (generated_text[:, prompt_len:-1] == termination_token_id).cumsum(-1) >= 1

    batch_loss = 0.0
    total_lambda = 0.0
    generated_len = generated_text.shape[1] - prompt_len
    
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
    print(batch_loss)
    return batch_loss

