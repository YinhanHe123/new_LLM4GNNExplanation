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
        messages = [{"role":"system", "content": "You are an assistent in a research lab, helping to discover new drugs to treat AIDS"}]
        messages.extend(conversation.messages)
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )
        conversation.mark_processed()
        conversation.append_response(response.choices[0].message.content)
    return conversations

def get_CF_text(conversations: List[Conversation]):
    CF_text = []
    for conversation in conversations:
        if ":" in conversation.messages[-1]['content']: # returns in the format "old functional group: new functional group" for feedback query
            func_groups = conversation.messages[-1]['content'].split(":")
            old_substring = func_groups[0].strip().lower()
            new_substring = func_groups[1].strip().lower()
        else:
            old_substring = "__"
            new_substring = re.sub(r'\.', '', conversation.messages[-1]['content']).lower() # returns only the replacement functional group for cf query
        caption = re.sub(r'"', '', conversation.past_user_inputs[-1].split('. ')[-1])
        cf = caption.replace(old_substring, new_substring)
        CF_text.append(cf)
    return CF_text

def get_feedback(
    outputs,
    conversations: List[Conversation], 
    feedback_prompt: str,
    original_captions: List[str],
    dataset: str):
    for i, conversation in enumerate(conversations):
        conversation.add_user_input(feedback_prompt.format(
            smiles=outputs.SMILES[i],
            true_prob = outputs.true_prob[i],
            original_caption = original_captions[i],
            likely = "increase" if "increase" in conversation.past_user_inputs[-1] else "decrease",
            dataset_description = DATASET_QUERY_MAP[dataset][0],
            molecule_description = DATASET_QUERY_MAP[dataset][1],
        ))
    return conversations

def get_feasible_cf(orginal_cfs, max_num_nodes):
    feasible_cf_list = []
    for idx in range(len(orginal_cfs)):
        conversation = [Conversation(check_valid_query.format(orginal_cfs[idx]['cf']))]
        conversation = get_responses(conversation)
        if "INVALID" in conversation[0].generated_responses[-1]:
            conversation[0].add_user_input(get_valid_query.format(orginal_cfs[idx]['cf']))
            conversation = get_responses(conversation)
            orginal_cfs[idx]['cf'] = conversation[0].generated_responses[-1]
        
        cf_graph = smiles_to_graph(orginal_cfs[idx]['cf'], max_num_nodes)
        cf_graph['original_graph_idx'] = orginal_cfs['graph_idx']
        feasible_cf_list.append(cf_graph)
    return feasible_cf_list

def train_gnn(args, gnn, gnn_train_loader, gnn_val_loader):
    print("Training GNN")
    optimizer = torch.optim.Adam(gnn.parameters(), lr=args.gnn_lr, weight_decay=args.gnn_weight_decay)
    gnn.train()
    gnn_start_train_time = time.time()

    for epoch in tqdm(range(args.gnn_epochs)):
        train_loss = 0
        train_acc = 0
        for batch in gnn_train_loader:
            labels = batch['label'].to(args.device)
            optimizer.zero_grad()
            out = gnn(batch)
            loss = F.nll_loss(out,labels.long())
            loss.backward()
            optimizer.step()
            
            train_loss += loss
            train_acc += (out.argmax(dim=1) == labels).sum().item()
        
        if epoch % 10 == 0:
            with torch.no_grad():
                train_loss /= len(gnn_train_loader)
                train_acc /= len(gnn_train_loader.dataset)
                
                eval_loss = 0
                eval_acc = 0
                for batch in gnn_val_loader:
                    labels = batch['label'].to(args.device)
                    out = gnn(batch)
                    eval_loss += F.nll_loss(out,labels.long()).item()
                    eval_acc += (out.argmax(dim=1) == labels).sum().item()
                eval_loss /= len(gnn_val_loader)
                eval_acc /= len(gnn_val_loader.dataset)
                
            time_checkpoint = time.time()
            time_comsumed = time_checkpoint - gnn_start_train_time
            print(f'epoch: {epoch} | train_loss: {train_loss} | train_acc : {train_acc} | eval_loss: {eval_loss} | eval_acc: {eval_acc} | time_consumed: {time_comsumed}')
    return gnn

def test_gnn(gnn, gnn_test_loader):
    print("Testing GNN")
    test_loss = 0
    test_acc = 0
    for batch in gnn_test_loader:
        labels = batch['label'].to(gnn.device)
        out = gnn(batch)
        test_loss += F.nll_loss(out,labels.long()).item()
        test_acc += (out.argmax(dim=1) == labels).sum().item()
    test_loss /= len(gnn_test_loader)
    test_acc /= len(gnn_test_loader.dataset)
    print(f'test_loss: {test_loss} | test_acc : {test_acc}')

def pretrain_autoencoder(args, autoencoder, train_loader, val_loader):
    optimizer = torch.optim.AdamW(
        [param for name, param in autoencoder.named_parameters() if 'decoder' not in name],
        lr=args.exp_pretrain_lr,
        weight_decay=args.exp_pretrain_weight_decay)
    autoencoder.train()
    
    for epoch in range(args.exp_pretrain_epochs):
        train_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = autoencoder.forward_clip(batch, return_loss=True)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            
            train_loss += loss
        if epoch % 10 == 0:
            with torch.no_grad():
                train_loss /= len(train_loader)
                eval_loss = 0
                for batch in tqdm(val_loader):
                    outputs = autoencoder.forward_clip(batch, return_loss=True)
                    loss = outputs['loss']
                    eval_loss += loss
                eval_loss /= len(val_loader)
            print(f'epoch: {epoch} | train_loss: {train_loss} | eval_loss: {eval_loss}')

def train_autoencoder(args, autoencoder, train_loader):
    final_outputs = []
    optimizer = torch.optim.AdamW(
            [param for _, param in autoencoder.named_parameters()],
            lr=args.exp_train_lr,
            weight_decay=args.exp_train_weight_decay)
    if args.ablation_type == 'nt':
        autoencoder.eval()
    else:
        autoencoder.train()

    for epoch in range(args.exp_train_epochs):
        feedback_times = 1 if args.ablation_type == "nf" else args.exp_feedback_times
        train_steps_in_one_feedback = 1 if args.ablation_type == "nf" else args.exp_train_steps_per_feedback
        train_loss = 0
        for batch in tqdm(train_loader):
            for _ in range(feedback_times):
                conversations = get_responses(conversations) # get CF responses
                CF_text = get_CF_text(conversations)
                for _ in range(train_steps_in_one_feedback):
                    optimizer.zero_grad()
                    outputs = autoencoder.forward_CF(batch, CF_text=CF_text, return_loss=True)
                    loss = outputs.loss
                    loss.backward()
                    train_loss += loss
                    optimizer.step()                        
                conversations = get_feedback(outputs, conversations, feedback_format, CF_text, train_loader.dataset.dataset)
        if epoch % 10 == 0:
            with torch.no_grad():
                train_loss /= (len(train_loader) * train_steps_in_one_feedback * feedback_times)
                print(f'epoch: {epoch} | train_loss: {train_loss}')

        if epoch == args.exp_train_epochs - 1:
            final_outputs.append({"graph_idx" : idx, "cf": smiles} for idx in batch['graph_idx'] for smiles in outputs.SMILES)
    
    return autoencoder, final_outputs


def compute_validity(cf_list, gt_gnn):
    for cf in cf_list:
        cf_preds = gt_gnn(cf).argmax(dim=1)
    percent_correct = torch.count_nonzero(cf_preds).item() / cf_preds.shape[-1] * 100

    valid_cf_list = []
    for cf in cf_list:
        temp_mol = Chem.MolFromSmiles(cf['smiles'])
        if temp_mol != None:
            valid_cf_list.append(cf)
    return percent_correct, valid_cf_list 

def compute_proximity(valid_cf_list, dataset):
    dist = 0
    for cf in valid_cf_list:
        dist += torch.cdist(dataset.__getitem__(cf['original_graph_idx'])['adj'], cf['adj'])
    return dist / len(valid_cf_list)