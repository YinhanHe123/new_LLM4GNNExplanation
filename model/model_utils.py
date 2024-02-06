from math import ceil
from symbol import atom
import time
from typing import List
from tqdm import tqdm
from transformers import Conversation
from utils.pre_defined import *
from utils.smiles import *
import openai
import re
import torch
import torch.nn.functional as F
from rdkit import Chem

openai.api_key = openai_api_key

MODEL_PRETRAIN_MAP = {"Bert": "bert-base-uncased"}

def get_responses(conversations: List[Conversation]):
    for conversation in conversations:
        messages = [{"role":"system", "content": "You are an assistent in a chemical research lab, helping to discover new molecules"}]
        messages.extend(conversation.messages)
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            temperature=0.2
        )
        conversation.mark_processed()
        conversation.append_response(response.choices[0].message.content)
    return conversations

def get_CF_text(conversations: List[Conversation]):
    CF_text = []
    for conversation in conversations:
        if "sorry" in conversation.messages[-1]['content'].lower():
            old_substring = ""
            new_substring = ""
        elif ":" in conversation.messages[-1]['content']: # returns in the format "old functional group: new functional group" for feedback query
            func_groups = conversation.messages[-1]['content'].split(":")
            old_substring = func_groups[0].strip().lower()
            new_substring = func_groups[1].strip().lower()
        else:
            old_substring = "__"
            new_substring = re.sub(r'\.', '', conversation.messages[-1]['content']).lower().strip() # returns only the replacement functional group for cf query
        caption = re.sub(r'"', '', conversation.past_user_inputs[-1].split('. ')[-1])
        cf = caption.replace(old_substring, new_substring)
        CF_text.append(cf)
    return CF_text

def get_feedback(
    outputs,
    conversations: List[Conversation], 
    original_captions: List[str],
    dataset: str):
    for i, conversation in enumerate(conversations):
        conversation.add_user_input(feedback_format.format(
            smiles=outputs.SMILES[i],
            true_prob = outputs.true_prob[i],
            original_caption = original_captions[i],
            likely = "increase",
            dataset_description = DATASET_QUERY_MAP[dataset][0],
            molecule_description = DATASET_QUERY_MAP[dataset][1],
        ))
    return conversations

def get_feasible_cf(original_cfs, max_num_nodes, dataset):
    feasible_cf_list = []
    for idx in range(len(original_cfs)):
        atom_features, adjacency_matrix, edge_attr_matrix, mask = smiles_to_graph(original_cfs[idx]['cf'], dataset.dataset, max_num_nodes)
        if atom_features is None:
            original_smiles = dataset.__getitem__(original_cfs[idx]['graph_idx'].detach().clone())
            conversation = [Conversation(reconst_query.format(cf_smiles=original_cfs[idx]['cf'], original_smiles = original_smiles, max_num_nodes=max_num_nodes))]
            conversation = get_responses(conversation)
            cf = re.search(r'"(.*?)"', conversation[0].generated_responses[-1], re.IGNORECASE)
            if cf is not None:
                original_cfs[idx]['cf'] = cf.group(1)
                try:
                    atom_features, adjacency_matrix, edge_attr_matrix, mask = smiles_to_graph(original_cfs[idx]['cf'], dataset.dataset, max_num_nodes)
                except IndexError:
                    pass
        
        cf_graph = {'x': atom_features, 'edge_attr': edge_attr_matrix, 'adj': adjacency_matrix, 'mask': mask, 
                    'smiles': original_cfs[idx]['cf'],'graph_idx': original_cfs[idx]['graph_idx'].item(), 'true_prob': original_cfs[idx]['true_prob']}
        feasible_cf_list.append(cf_graph)
    return feasible_cf_list

def pretrain_autoencoder(args, autoencoder, train_loader, val_loader):
    optimizer = torch.optim.AdamW(
        [param for name, param in autoencoder.named_parameters() if 'decoder' not in name],
        lr=args.exp_pretrain_lr,
        weight_decay=args.exp_pretrain_weight_decay)
    autoencoder.train()
    autoencoder.graph_encoder.eval()
    
    for epoch in range(args.exp_pretrain_epochs):
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = autoencoder.forward_clip(batch, return_loss=True)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            
            train_loss += loss
        if epoch % 10 == 0 or epoch == args.exp_pretrain_epochs - 1:
            with torch.no_grad():
                train_loss /= len(train_loader)
                eval_loss = 0
                for batch in val_loader:
                    outputs = autoencoder.forward_clip(batch, return_loss=True)
                    loss = outputs['loss']
                    eval_loss += loss
                eval_loss /= len(val_loader)
            print(f'pretrain epoch: {epoch} | train_loss: {train_loss} | eval_loss: {eval_loss}')
    return autoencoder

def train_autoencoder(args, autoencoder, train_loader, val_loader):
    optimizer = torch.optim.AdamW(
            [param for _, param in autoencoder.named_parameters()],
            lr=args.exp_train_lr,
            weight_decay=args.exp_train_weight_decay)
    if args.ablation_type == 'nt':
        autoencoder.eval()
    else:
        autoencoder.train()
    autoencoder.graph_encoder.eval()
    
    feedback_times = 1 if args.ablation_type == "nf" else args.exp_feedback_times
    train_steps_in_one_feedback = 1 if args.ablation_type == "nf" else args.exp_train_steps_per_feedback
    num_batches_per_epoch = ceil(args.exp_data_percent * len(train_loader)) \
                    if ceil(args.exp_data_percent * len(train_loader)) > 10 else min(len(train_loader), 30)
    
    for epoch in range(args.exp_train_epochs):
        train_loss = 0
        selected_idxs = np.random.choice(range(len(train_loader)), size=num_batches_per_epoch, replace=False)
        for idx, batch in enumerate(tqdm(train_loader)):
            if idx not in selected_idxs:
                continue
            batch = next(iter(train_loader))
            conversations = [Conversation(cf_query) for cf_query in batch['cf_query']]
            for i in range(feedback_times):
                conversations = get_responses(conversations) # get CF responses
                print('Sample conversation for ' + str(i) + ' feedback' + ':\n', conversations[0])
                CF_text = get_CF_text(conversations)
                for _ in range(train_steps_in_one_feedback):
                    optimizer.zero_grad()
                    outputs = autoencoder.forward_CF(batch, CF_text=CF_text, return_loss=True)
                    loss = outputs.loss
                    loss.backward()
                    train_loss += loss
                    optimizer.step()
                conversations = get_feedback(outputs, conversations, CF_text, train_loader.dataset.dataset)
        with torch.no_grad():
            train_loss /= (num_batches_per_epoch * train_steps_in_one_feedback * feedback_times)
            # test autoencoder in validation set
            eval_loss = 0
            for batch in val_loader:
                outputs = autoencoder.forward_CF(batch, CF_text=batch['cf_query'], return_loss=True)
                loss = outputs.loss
                eval_loss += loss
            eval_loss /= len(val_loader)              
            print(f'Train autoencoder restart round: {epoch} | train_loss: {train_loss} | val_loss: {eval_loss}')
    return autoencoder
