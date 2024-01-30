import torch
import argparse
from tqdm import tqdm
import os
from transformers import AutoModel, AutoConfig, AutoTokenizer

from utils.pre_defined import *
from utils.smiles import *
from utils.datasets import Dataset
from model.gnn import GCN
from model.explainer import GraphTextClipModel
from model.model_utils import *

def parse_llm_gce_args():
    parser = argparse.ArgumentParser("LLM Guided Global Counterfactual Explanations")

    parser.add_argument("-d", "--dataset", type=str, default='AIDS')
    parser.add_argument("--train_ratio", default=0.6, type=float)
    parser.add_argument("--val_ratio", default=0.2, type=float)
    parser.add_argument("--splits", default='random', type=str, help="The split datasets way")
    parser.add_argument("--device", default=0, type=int, help='GPU device')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("-bsz", "--batch_size", default=32, type=int)
    parser.add_argument("-nat","--num_atom_types", default=len(NODE_LABEL_MAP), type=int, help='num atom types')
    parser.add_argument("-at","--ablation_type", default=None, choices=['np', 'nt', 'nf'],
                help='np - no pretrained text encoder, nt - no training of text encoder, nf - no feedback')
    
    # GNN args
    parser.add_argument("-ge", "--gnn_epochs", default=100, type=int)
    parser.add_argument("-glr", "--gnn_lr", default=0.005, type=float, help='GNN model learning rate')
    parser.add_argument("-gwd", "--gnn_weight_decay", default=0.01, type=float, help='GNN model weight decay')

    # Autoencoder args
    parser.add_argument("-lmm", "--lm_model", default="Bert", type=str, help='LM model used in the autoencoder')
    parser.add_argument("-hd", "--h_dim", default=512, type=int, help='autoencoder hidden dime')
    parser.add_argument("-mcl", "--max_context_length", default=4096, type=int, help='max context length')
    parser.add_argument("-mm", "--m_mu", default=1.0, type=float)
    parser.add_argument("-cm", "--c_mu", default=0.5, type=float)
    parser.add_argument("-epe", "--exp_pretrain_epochs", default=10, type=int)
    parser.add_argument("-eplr", "--exp_pretrain_lr", default=0.002, type=float)
    parser.add_argument("-epwd", "--exp_pretrain_weight_decay", default=1e-5, type=float)
    parser.add_argument("-ete", "--exp_train_epochs", default=10, type=int)
    parser.add_argument("-etlr", "--exp_train_lr", default=0.002, type=float)
    parser.add_argument("-etwd", "--exp_train_weight_decay", default=1e-5, type=float)
    parser.add_argument("-eft", "--exp_feedback_times", default=3, type=int)
    parser.add_argument("-etpf", "--exp_train_steps_per_feedback", default=20, type=int)

    return parser.parse_args()

def main():
    args = parse_llm_gce_args()
    args.device='cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu'
    
    # Load data
    dataset = Dataset(dataset=args.dataset)
    data_split = [0.5, 0.25, 0.25]
    
    gnn_train_loader, gnn_val_loader, gnn_test_loader = dataset.get_dataloaders(args.batch_size, data_split)
    pretrain_train_loader, pretrain_val_loader, _ = dataset.get_dataloaders(args.batch_size, data_split, mask_pos = True)
    # smaller batch size for contexts management
    explainer_train_loader, _, _ =  dataset.get_dataloaders(4, data_split, mask_pos = True)
    
    # Get ground truth GNN
    gnn = GCN(num_classes=2, num_features=args.num_atom_types, embedding_dim=128, device=args.device).to(args.device)
    gnn_path = './saved_models/gnn_'+args.dataset+'.pth'
    if os.path.isfile(gnn_path):
        gnn.load_state_dict(torch.load(gnn_path))
    else:
        gnn = train_gnn(args, gnn, gnn_train_loader, gnn_val_loader)
        test_gnn(gnn, gnn_test_loader)
        torch.save(gnn.state_dict(), gnn_path)
        print('gnn model saved to {}'.format(gnn_path))
    gnn.eval()

    # Load autoencoder
    explainer = GraphTextClipModel(
            text_encoder=AutoModel.from_pretrained(MODEL_PRETRAIN_MAP[args.lm_model]),
            tokenizer=AutoTokenizer.from_pretrained(MODEL_PRETRAIN_MAP[args.lm_model]),
            lmconfig=AutoConfig.from_pretrained(MODEL_PRETRAIN_MAP[args.lm_model]),
            graph_encoder=gnn,
            args=args,
            max_num_nodes=dataset.max_num_nodes,
        ).to(args.device)
    
    # pretrain
    if args.ablation_type != "np":
        pretrain_lm_path = './LMs/SavedPretrainedLMs/'+args.dataset+'.pth'
        if os.path.isfile(pretrain_lm_path):
            explainer.load_state_dict(torch.load(pretrain_lm_path))
        else:
            explainer = pretrain_autoencoder(args, explainer, pretrain_train_loader, pretrain_val_loader)
            torch.save(explainer.state_dict(), pretrain_lm_path)

    # training
    explainer, final_outputs = train_autoencoder(args, explainer, explainer_train_loader)

    """
    Alternate for 159-185:
    # for each epoch: (here, the total epochs is the train_steps_in_one_feedback * feedback_times)
        for each batch:
            autoencoder_update
            if epoch % train_steps_in_one_feedback == 0:
                CTA_update
    get the counterfactual graphs
    test both (run with restart/ multiple epochs vs continuing the session)
    """
    
    # chemical faesibility adjustment.
    feasible_cf_list = get_feasible_cf(final_outputs, dataset.max_num_nodes)

    # evaluate cfs    
    validity, valid_cf_list = compute_validity(feasible_cf_list, gnn)
    proximity = compute_proximity(valid_cf_list, dataset)
    print("validity: ", validity)
    print("proximity: ", proximity)
    # TODO: Store those evaluation results into a file.


    # Evaluate:
    # 1. Validity:
    # 1.1 Put the counterfactuals into GT-GNN, see how many of them are classified as desired.
    # 1.2 Put the desired counterfactuals into the SMILES feasibility checker funtion to see if it satisfy valence theory.
    # 2. Proximity:
    # 2.1 calculate all the MEAN VALUE of the graph distance of the counterfactuals valid in sense of 1.2 and their original counterparts.
    
if '__main__' == __name__:
    main()
    