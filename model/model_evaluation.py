import torch
from model.model_utils import *

def evaluate_gce_model(test_loader, gcn, dataset, explainer):
    """
    Evaluate the GCE model.
    """
    final_outputs = []        
    for batch in test_loader:
        outputs = explainer.forward_CF(batch, CF_text=batch['cf_query'], return_loss=True)
        final_outputs.extend([{"graph_idx" : idx, "cf": smiles, 'true_prob': true_prob} for idx, smiles, true_prob in zip(batch['graph_idx'], outputs.SMILES, outputs.true_prob)])
    feasible_cf_list = get_feasible_cf(final_outputs, dataset.max_num_nodes, dataset)
    validity, valid_cf_list = compute_validity(feasible_cf_list, gcn)
    proximity = compute_proximity(valid_cf_list, dataset)
    return validity, proximity


def compute_validity(cf_list, gt_gnn):
    cf_preds = []
    valid_cf_list = []
    for cf in cf_list:
        if cf['x'] is not None: 
            if cf['true_prob']>0.5:
                valid_cf_list.append(cf)  
            cf_preds.append(cf['true_prob'] > 0.5)
    percent_correct_and_valid = sum(cf_preds) / len(cf_list) * 100
    return percent_correct_and_valid, valid_cf_list 

def compute_proximity(valid_cf_list, dataset):
    dist = 0
    for cf in valid_cf_list:
        # torch.sqrt(torch.sum((dataset.__getitem__(cf['graph_idx'])['adj'] - cf['adj']).pow(2)))
        dist += torch.sqrt(torch.sum((dataset.__getitem__(cf['graph_idx'])['adj'] - cf['adj']).pow(2))).item()
    if len(valid_cf_list) == 0:
        return 0
    return dist / len(valid_cf_list)
