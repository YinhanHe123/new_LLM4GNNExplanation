import pandas as pd
import torch
from model.model_utils import *

def evaluate_gce_model(test_loader, gcn, dataset, explainer):
    """
    Evaluate the GCE model.
    """
    print('----------------------Testing Autoencoder----------------------\n')
    final_outputs = []
    batch = next(iter(test_loader))
    for batch in tqdm(test_loader):
        outputs = explainer.forward_CF(batch, CF_text=batch['cf_query'], return_loss=True)
        for idx, smiles, true_prob, x_reconst, adj_reconst, edge_reconst in zip(batch['graph_idx'], outputs.SMILES, outputs.true_prob, outputs.x_reconst, outputs.adj_reconst, outputs.edge_reconst):
            x_reconst = x_reconst.detach().cpu().argmax(dim=-1)
            adj_reconst = (adj_reconst.detach().cpu() > 0.5).to(float)
            edge_reconst = torch.round(edge_reconst.detach().cpu())
            cf = {"graph_idx" : idx, "cf": smiles, 'true_prob': true_prob, 'x': x_reconst, 'adj': adj_reconst, 'edge_attr': edge_reconst}
            final_outputs.append(cf)
            
    feasible_cf_list = get_feasible_cf(final_outputs, dataset.max_num_nodes, dataset)

    validity_without_chem, true_cf_list = compute_validity_without_chem(final_outputs)
    proximity_without_chem = compute_proximity_without_chem(true_cf_list, dataset)

    validity, valid_cf_list = compute_validity(feasible_cf_list, gcn)
    proximity = compute_proximity(valid_cf_list, dataset)

    original_cfs = [x['cf'] for x in final_outputs]
    true_cfs = [x in true_cf_list for x in final_outputs]
    feasible_cfs = [x['smiles'] for x in feasible_cf_list]
    valid_feasible_cfs = [x in valid_cf_list for x in feasible_cf_list]
    results = pd.DataFrame(
    {'original_cfs': original_cfs,
     'true_cf': true_cfs,
     'improved_cfs': feasible_cfs,
     'feasible_cf': valid_feasible_cfs
    })

    return validity, proximity, validity_without_chem, proximity_without_chem, results


def compute_validity(cf_list, gt_gnn):
    cf_preds = []
    valid_cf_list = []
    for cf in cf_list:
        if cf['x'] is not None: 
            cf['x'] = cf['x'].unsqueeze(0)
            cf['adj'] = cf['adj'].unsqueeze(0)
            cf['edge_attr'] = cf['edge_attr'].unsqueeze(0)
            cf['mask'] = cf['mask'].unsqueeze(0)
            cf['true_prob'] = torch.exp(gt_gnn(cf)['y_pred'])[:, 1].item()
            if cf['true_prob']>0.5:
                valid_cf_list.append(cf)  
            cf_preds.append(cf['true_prob'] > 0.5)
    percent_correct_and_valid = sum(cf_preds) / len(cf_list) * 100
    return percent_correct_and_valid, valid_cf_list 

def compute_validity_without_chem(final_outputs):
    true_cf_list = []
    for cf in final_outputs:
        if cf['true_prob']>0.5:
            true_cf_list.append(cf)
    percent_correct = len(true_cf_list) / len(final_outputs) * 100
    return percent_correct, true_cf_list 

def compute_proximity_without_chem(true_cf_list, dataset):
    dist = 0
    for cf in true_cf_list:
        dist += torch.sqrt(torch.sum((dataset.__getitem__(cf['graph_idx'])['adj'] - cf['adj']).pow(2))).item()+\
                torch.sqrt(torch.sum((dataset.__getitem__(cf['graph_idx'])['x'] - cf['x']).pow(2))).item()+\
                torch.sqrt(torch.sum((dataset.__getitem__(cf['graph_idx'])['edge_attr'] - cf['edge_attr']).pow(2))).item()
    if len(true_cf_list) == 0:
        return -1
    return dist / len(true_cf_list)

def compute_proximity(valid_cf_list, dataset):
    dist = 0
    for cf in valid_cf_list:
        # torch.sqrt(torch.sum((dataset.__getitem__(cf['graph_idx'])['adj'] - cf['adj']).pow(2)))
        dist += torch.sqrt(torch.sum((dataset.__getitem__(cf['graph_idx'])['adj'] - cf['adj']).pow(2))).item()+\
                torch.sqrt(torch.sum((dataset.__getitem__(cf['graph_idx'])['x'] - cf['x']).pow(2))).item()+\
                torch.sqrt(torch.sum((dataset.__getitem__(cf['graph_idx'])['edge_attr'] - cf['edge_attr']).pow(2))).item()
    if len(valid_cf_list) == 0:
        return -1
    return dist / len(valid_cf_list)
