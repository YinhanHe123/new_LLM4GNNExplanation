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
def response_stub(messages):
    user_input = messages[-1]['content']
    if "__" in user_input:
        return "methyl group"
    elif "VALID" in user_input:
        return "VALID"
    else:
        old_component = re.search(r'(in which |where |of which )(the )*(.*?) may', user_input, re.IGNORECASE).group(3)
        return old_component + ": methyl group"


def get_responses(conversations: List[Conversation]):
    for conversation in conversations:
        messages = [{"role":"system", "content": "You are an assistent in a research lab, helping to discover new drugs to treat AIDS"}]
        messages.extend(conversation.messages)
        # response = openai.chat.completions.create(
        #     model="gpt-3.5-turbo-1106",
        #     messages=messages,
        #     temperature=0.3
        # )
        # conversation.mark_processed()
        # conversation.append_response(response.choices[0].message.content)
        conversation.mark_processed()
        conversation.append_response(response_stub(messages))
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

def get_feasible_cf(orginal_cfs, max_num_nodes, dataset):
    feasible_cf_list = []
    for idx in range(len(orginal_cfs)):
        conversation = [Conversation(check_valid_query.format(molecule=orginal_cfs[idx]['cf']))]
        conversation = get_responses(conversation)
        if "INVALID" in conversation[0].generated_responses[-1]:
            conversation[0].add_user_input(get_valid_query.format(molecule=orginal_cfs[idx]['cf']))
            conversation = get_responses(conversation)
            orginal_cfs[idx]['cf'] = conversation[0].generated_responses[-1]
        
        atom_features, adjacency_matrix, edge_attr_matrix, mask = smiles_to_graph(orginal_cfs[idx]['cf'], dataset, max_num_nodes)
        cf_graph = {'x': atom_features, 'edge_attr': edge_attr_matrix, 'adj': adjacency_matrix, 'mask': mask, 
                    'smiles': orginal_cfs[idx]['cf'],'graph_idx': orginal_cfs[idx]['graph_idx'].item()}
        feasible_cf_list.append(cf_graph)
    return feasible_cf_list

# def train_gnn(args, gnn, gnn_train_loader, gnn_val_loader):
#     print("Training GNN")
#     optimizer = torch.optim.Adam(gnn.parameters(), lr=args.gnn_lr, weight_decay=args.gnn_weight_decay)
#     gnn.train()
#     gnn_start_train_time = time.time()

#     for epoch in tqdm(range(args.gnn_epochs)):
#         train_loss = 0
#         train_acc = 0
#         for batch in gnn_train_loader:
#             labels = batch['label'].to(args.device)
#             optimizer.zero_grad()
#             out = gnn(batch)
#             out = out['y_pred']
#             loss = F.nll_loss(out,labels.long())
#             loss.backward()
#             optimizer.step()
            
#             train_loss += loss
#             train_acc += (out.argmax(dim=1) == labels).sum().item()
        
#         if epoch % 10 == 0 or epoch == args.gnn_epochs - 1:
#             with torch.no_grad():
#                 train_loss /= len(gnn_train_loader)
#                 train_acc /= len(gnn_train_loader.sampler)
                
#                 eval_loss = 0
#                 eval_acc = 0
#                 for batch in gnn_val_loader:
#                     labels = batch['label'].to(args.device)
#                     out = gnn(batch)
#                     out = out['y_pred']
#                     eval_loss += F.nll_loss(out,labels.long()).item()
#                     eval_acc += (out.argmax(dim=1) == labels).sum().item()
#                 eval_loss /= len(gnn_val_loader)
#                 eval_acc /= len(gnn_val_loader.sampler)
                
#             time_checkpoint = time.time()
#             time_comsumed = time_checkpoint - gnn_start_train_time
#             print(f'epoch: {epoch} | train_loss: {train_loss} | train_acc : {train_acc} | eval_loss: {eval_loss} | eval_acc: {eval_acc} | time_consumed: {time_comsumed}')
    
#     return gnn

# def test_gnn(gnn, gnn_test_loader):
#     print("Testing GNN")
#     test_loss = 0
#     test_acc = 0
#     for batch in gnn_test_loader:
#         labels = batch['label'].to(gnn.device)
#         out = gnn(batch)
#         out = out['y_pred']
#         test_loss += F.nll_loss(out,labels.long()).item()
#         test_acc += (out.argmax(dim=1) == labels).sum().item()
#     test_loss /= len(gnn_test_loader)
#     test_acc /= len(gnn_test_loader.sampler)
#     print(f'test_loss: {test_loss} | test_acc : {test_acc}')

def pretrain_autoencoder(args, autoencoder, train_loader, val_loader):
    optimizer = torch.optim.AdamW(
        [param for name, param in autoencoder.named_parameters() if 'decoder' not in name],
        lr=args.exp_pretrain_lr,
        weight_decay=args.exp_pretrain_weight_decay)
    autoencoder.train()
    
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
                for batch in tqdm(val_loader):
                    outputs = autoencoder.forward_clip(batch, return_loss=True)
                    loss = outputs['loss']
                    eval_loss += loss
                eval_loss /= len(val_loader)
            print(f'pretrain epoch: {epoch} | train_loss: {train_loss} | eval_loss: {eval_loss}')
    return autoencoder

def train_autoencoder(args, autoencoder, train_loader, val_loader):
    final_outputs = []
    optimizer = torch.optim.AdamW(
            [param for _, param in autoencoder.named_parameters()],
            lr=args.exp_train_lr,
            weight_decay=args.exp_train_weight_decay)
    if args.ablation_type == 'nt':
        autoencoder.eval()
    else:
        autoencoder.train()
    
    feedback_times = 1 if args.ablation_type == "nf" else args.exp_feedback_times
    train_steps_in_one_feedback = 1 if args.ablation_type == "nf" else args.exp_train_steps_per_feedback
    
    for epoch in range(args.exp_train_epochs):
            train_loss = 0
            batch = next(iter(train_loader))
        # for batch in tqdm(train_loader):
            conversations = [Conversation(cf_query) for cf_query in batch['cf_query']]
            # unfinished_conversations = conversations.copy()
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
                # TODO:add those graph indices that have counterfactual probability higher than 0.8, renew conversations
            with torch.no_grad():
                train_loss /= (len(train_loader) * train_steps_in_one_feedback * feedback_times)
                # test autoencoder in validation set
                eval_loss = 0
                for batch in val_loader:
                    outputs = autoencoder.forward_CF(batch, CF_text=batch['cf_query'], return_loss=True)
                    loss = outputs.loss
                    eval_loss += loss
                eval_loss /= len(val_loader)              
                print(f'Train autoencoder restart round: {epoch} | train_loss: {train_loss} | val_loss: {eval_loss}')
            if epoch == args.exp_train_epochs - 1:
                final_outputs.extend([{"graph_idx" : idx, "cf": smiles} for idx, smiles in zip(batch['graph_idx'], outputs.SMILES)])
    return autoencoder, final_outputs
