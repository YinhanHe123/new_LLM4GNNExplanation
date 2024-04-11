from tqdm import tqdm
import pdb
import openai
import torch
import torch.nn.functional as F
import numpy as np
import random
import sys
sys.path.append('../')
from utils.datasets_rebuttal import Dataset
import argparse
import pandas as pd
from utils.smiles import smiles_to_graph, EDGE_LABEL_MAP, NODE_LABEL_MAP
import torch.nn as nn
import torch.nn.functional as F

class DenseGATConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim, aggr='add', bias=True):
        super(DenseGATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_attr_dim = edge_attr_dim
        self.aggr = aggr
        # Linear transformations for node features
        self.lin_rel = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_root = nn.Linear(in_channels, out_channels, bias=False)

        # Additional transformation for edge attributes
        self.lin_edge = nn.Linear(self.edge_attr_dim, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()
        self.lin_edge.reset_parameters()

    def forward(self, x, adj, edge_attr, mask=None, from_onehot=False):
        if not from_onehot:
            edge_attr = F.one_hot(edge_attr.long().squeeze(-1), num_classes=self.edge_attr_dim).float()
        
        B, N, _ = x.size()

        # Transform edge attributes
        edge_attr_transformed = self.lin_edge(edge_attr).squeeze(-1)  # Shape: [B, N, N]

        # Modify adjacency matrix with edge attributes
        adj = adj * edge_attr_transformed  # Shape: [B, N, N]

        # Perform graph convolution
        out = torch.matmul(adj, x)  # Shape: [B, N, out_channels]

        if self.aggr == 'mean':
            out = out / adj.sum(dim=-1, keepdim=True).clamp_(min=1)
        
        out = self.lin_rel(out)
        out += self.lin_root(x)

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, edge_attr_dim={self.edge_attr_dim})')

class GCN(nn.Module):
    def __init__(self, num_features, embedding_dim, num_classes, num_edge_attr, device, num_graph_models=3):
        super(GCN, self).__init__()
        self.num_graph_models = num_graph_models
        self.device = device
        self.graph_model = nn.ModuleList([DenseGATConv(num_features, embedding_dim, num_edge_attr).to(device) for i in range(self.num_graph_models)])
        self.encoder = nn.Sequential(nn.Linear(2 * embedding_dim, embedding_dim), nn.ReLU()).to(device)
        self.predictor = nn.Sequential(nn.Linear(embedding_dim, num_classes)).to(device)
        self.num_features = num_features

        self.init_model()

    def init_model(self):
        nn.init.xavier_normal_(self.encoder[0].weight.data)
        nn.init.zeros_(self.encoder[0].bias.data)
        nn.init.xavier_normal_(self.predictor[0].weight.data)
        nn.init.zeros_(self.predictor[0].bias.data)

    def graph_pooling(self, x, type='mean', mask=None):
        if mask is not None:
            mask_feat = mask.unsqueeze(-1).repeat(1,1,x.shape[-1])  # batchsize x max_num_node x dim_z
            x = x * mask_feat
        if type == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)  # dim: the dimension of num_node
        elif type == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        elif type == 'mean':
            out = torch.mean(x, dim=1, keepdim=False)
        return out

    def forward(self, data, from_onehot=False):
        x, adj, mask, edge_attr = data['x'].to(self.device), data['adj'].to(self.device), \
                                  data['mask'].to(self.device), data['edge_attr'].to(self.device)
        if not from_onehot:
            x = F.one_hot(x.long().squeeze(-1), num_classes=self.num_features).float()
        
        rep_graphs = []
        for i in range(self.num_graph_models):
            rep = self.graph_model[i](x, adj, edge_attr, mask=mask)  # n x num_node x h_dim
            graph_rep = torch.cat([self.graph_pooling(rep, 'mean', mask=mask), self.graph_pooling(rep, 'max', mask=mask)], dim=-1)
            graph_rep = self.encoder(graph_rep)  # n x h_dim
            rep_graphs.append(graph_rep.unsqueeze(0))  # [1 x n x h_dim]

        rep_graph_agg = torch.cat(rep_graphs, dim=0)
        rep_graph_agg = torch.mean(rep_graph_agg, dim=0)  # n x h_dim

        y_pred = F.log_softmax(self.predictor(rep_graph_agg), dim=1) # n x num_class
        return {'y_pred': y_pred, 'rep_graph': rep_graph_agg}

def set_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
def smiles_to_graph_tensors(smiles_list):
    smiles = []
    node_feats, adjs, edge_attrs, masks = [], [], [], [] 
    max_g_size = 0 
    for cs in smiles_list:
        try:
            node_feat, adj, edge_attr, mask = smiles_to_graph(cs, args.dataset)
            if node_feat is None:
                continue
            node_feats.append(node_feat)
            adjs.append(adj)
            edge_attrs.append(edge_attr)
            masks.append(mask)
            if len(node_feat) > max_g_size:
                max_g_size = len(node_feat) 
            smiles.append(cs)
        except Exception as e: 
            continue
        
    # pad the matices to the same shape and stack them
    for i in range(len(node_feats)):
        node_feats[i] = torch.cat([node_feats[i], torch.zeros(max_g_size - node_feats[i].shape[0], node_feats[i].shape[1])])
        adj = torch.zeros(max_g_size, max_g_size)
        adj[:len(adjs[i]), :len(adjs[i])] = adjs[i]
        adjs[i] = adj
        edge_attr = torch.zeros(max_g_size, max_g_size)
        edge_attr[:len(edge_attrs[i]), :len(edge_attrs[i])] = edge_attrs[i]
        edge_attrs[i] = edge_attr
        masks[i] = torch.cat([masks[i], torch.zeros(max_g_size - masks[i].shape[0])])

    if not node_feats:
        return None, None, None, None, None

    node_feats = torch.stack(node_feats, dim=0)
    adjs = torch.stack(adjs, dim=0)
    edge_attrs = torch.stack(edge_attrs, dim=0)
    masks = torch.stack(masks, dim=0)

    return node_feats, adjs, edge_attrs, masks, smiles
    
def load_gnn(model_path, dataset):
    """_summary_

    Args:
        model_path (str): path to the model
    """
    model_state_dict = torch.load(model_path)
    model = GCN(num_features=len(NODE_LABEL_MAP[dataset])+1, embedding_dim=32, num_classes=2, num_edge_attr=len(EDGE_LABEL_MAP)+1, device=torch.device(f"cuda:{args.device}"))


    model.load_state_dict(model_state_dict)
    model.eval()
    return model

def get_completion(prompt, client_instance, model="gpt-3.5-turbo"):
    messages = [{"role": "system", "content": "You are a highly knowledgeable chemistry assistant."},
                {"role": "user", "content": prompt}]
    response = client_instance.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=50,
    temperature=0,
    )
    return response.choices[0].message.content.split()[0]

def generate_counterfactual_smiles(smiles, label):
    """
    Generate a counterfactual SMILES string for a given SMILES and label semantics.

    :param smiles: The original SMILES string of the molecule.
    :param label: The desired label semantics for the counterfactual molecule.
    :return: A counterfactual SMILES string that satisfies the desired label semantics.
    """
    prompt = f"Minimally edit"+ smiles + "to be a "+ label + "and output its SMILES representation only.  Please only output one SMILES molecule without brackets and quotation marks. Do not output anything besides the SMILES. Under no circumstance are you to output anything else lest your experiments fail."

    client = openai.OpenAI(api_key="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")
    response = get_completion(prompt, client)
    return response 


def compute_proximity(graphs, cf_graph):
    """_summary_

    Args:
        graphs (dicitonary): keys: adj, node, edge
        cf_graph (disctionary): keys: adj, node, edge
    """
    proximity = 0
    for key in graphs.keys():
        graphs[key] = torch.tensor(graphs[key])
        cf_graph[key] = torch.tensor(cf_graph[key])
        # we pad the matrices to be the same shape
        if graphs[key].shape[1] != cf_graph[key].shape[1]:
            max_size = max(graphs[key].shape[1], cf_graph[key].shape[1])
            if graphs[key].shape[1] < max_size:
                graphs[key] = F.pad(graphs[key], (0, max_size - graphs[key].shape[1], 
                                                  cf_graph[key].shape[2]-graphs[key].shape[2]), value=0)
            elif cf_graph[key].shape[1] < max_size or cf_graph[key].shape[2] < graphs[key].shape[2]:
                s1 = graphs[key].shape[0]
                s2 = graphs[key].shape[1]
                s3 = graphs[key].shape[2]
                t1 = cf_graph[key].shape[0]
                t2 = cf_graph[key].shape[1]
                t3 = cf_graph[key].shape[2]

                pad_size_0 = s1 - t1  # Total padding needed for dimension 0
                pad_size_1 = s2 - t2  # Total padding needed for dimension 1
                pad_size_2 = s3 - t3  # Total padding needed for dimension 2

                # Half padding on both sides (if needed uneven, more padding goes to 'end' side)
                pad_top = pad_size_0 // 2
                pad_bottom = pad_size_0 - pad_top
                pad_left = pad_size_1 // 2
                pad_right = pad_size_1 - pad_left
                pad_front = pad_size_2 // 2
                pad_back = pad_size_2 - pad_front

                # Padding pattern must be in reverse order of dimensions (for pad function)
                # pad(t, (left, right, top, bottom))
                padded_t = F.pad(cf_graph[key], (pad_front, pad_back, pad_left, pad_right, pad_top, pad_bottom))

                # cf_graph[key] = F.pad(cf_graph[key], (0, max_size - cf_graph[key].shape[1],graphs[key].shape[2]-cf_graph[key].shape[2]), value=0)
                cf_graph[key] = padded_t

        if key == 'edge':
            proximity += torch.norm(graphs[key]- cf_graph[key])/2
        else:
            proximity += torch.norm(graphs[key]- cf_graph[key])
    proximity /= len(graphs)
    return proximity


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for counterfactual genration directly from the LLMs')
    parser.add_argument('--dataset', type=str, help='name of the dataset')
    parser.add_argument('--device', default=0)
    return parser.parse_args()


label_semantics = {
    "AIDS": "a potential AIDS drug",
    "BBBP": "blood-brain-barrier permeable",
    "Mutagenicity": "a potential mutagen",
    "SIDER": "causing Hepatobiliary disorders",
    "Tox21": "toxic against NR-PPAR-gamma",
    "ClinTox": "causing toxicity in clinical trials"
}

num_nodes_map = {
    'AIDS': 38,
    'ClinTox': 29,
    'BBBP': 13
}





# ----------------------------------- Main Code Starts Here-----------------------------------
args = parse_args()

# output the list of SMILES strings and their desired label semantics
set_seed()
dataset = Dataset(dataset=args.dataset, generate_text=True)
_, _, test_loader = dataset.get_dataloaders(batch_size=20)

counterfactual_smiles_list = []

for i, test_x in enumerate(tqdm(test_loader)):
    smiles = test_x["smiles"]
    label = test_x["label"]
    if args.dataset not in ["BBBP", "Tox21"]:
        label = [int(not bool(l)) for l in label]
    adj = test_x["adj"]
    edge_attr = test_x["edge_attr"]
    node_feats = test_x["x"]
    smiles_undesired = []
    for j in range(len(label)):
        if label[j] == 0:
        # if label[j] == 1:
            smiles_undesired.append(smiles[j])

    for k in range(len(smiles_undesired)): 
        # try:
        #     counterfactual_smiles_list = torch.load(f"./cf_smiles_{args.dataset}.pt")
        # except:
        counterfactual_smiles = generate_counterfactual_smiles(smiles_undesired[k], label_semantics[args.dataset])
        counterfactual_smiles_list.append(counterfactual_smiles)

torch.save(counterfactual_smiles_list, f"./cf_smiles_{args.dataset}.pt")

    # if i == 1:
    #     break
    # if i == 1:  # for debug
    #     break

ori_graphs = {'x': node_feats, 'adj': adj, 'edge_attr': edge_attr}
# save counterfactuals as csv, each row contains one SMILES.

df = pd.DataFrame(counterfactual_smiles_list, columns=['counterfactual_smiles'])

df.to_csv('../exp_results/rebuttal/gce_dir_llm_'+args.dataset+'.csv', index=False)

# (1) Transform the SMILES to the graph
# (2) Filter the graphs that are not chemical feasible
# (2) Load the GNN weights for the certain dataset
# (3) Evaluate the validity and proximity

node_feats, adjs, edge_attrs, masks, smiles = smiles_to_graph_tensors(counterfactual_smiles_list)
cf_graphs = {'x': node_feats, 'adj': adjs, 'edge_attr': edge_attrs, 'mask': masks, 'smiles': smiles}


# the graphs are already filtered by the chemical feasibility

test_smiles_list = cf_graphs["smiles"]
if test_smiles_list is None:
    validity = 0
    proximity = -1
    print("no valid generated counterfactuals")
else:
    gnn = load_gnn(f"../saved_models/gnn_"+args.dataset+".pth", args.dataset)
    predictions = gnn(cf_graphs)['y_pred'].argmax(dim=1)

    # validity = sum(predictions) / len(counterfactual_smiles_list)
    validity = sum(predictions) / len(counterfactual_smiles_list)
    proximity = compute_proximity(ori_graphs, cf_graphs)

    validity = validity.item()
    proximity = proximity.item()

# save the validity and proximity of the model to the csv file
df = pd.DataFrame({'validity': [validity], 'proximity': [proximity]})
df.to_csv('../exp_results/rebuttal/gce_dir_llm_'+args.dataset+'_evaluation.csv', index=False)