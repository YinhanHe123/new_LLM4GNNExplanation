from datetime import datetime
import torch
import argparse
import os
from transformers import AutoModel, AutoConfig, AutoTokenizer
from model.model_evaluation import evaluate_gce_model
from utils.utils import write_file_start
from utils.pre_defined import *
from utils.smiles import *
from utils.datasets import Dataset
from model.gnn import gnn_trainer
from model.explainer import GraphTextClipModel
from model.model_utils import *

def parse_llm_gce_args():
    parser = argparse.ArgumentParser("LLM Guided Global Counterfactual Explanations")

    parser.add_argument("-ne", "--num_exps", default=5, type=int, help='number of experiments')

    parser.add_argument("-d", "--dataset", type=str, default='AIDS')
    parser.add_argument("--train_ratio", default=0.6, type=float, help='train ratio for both GT-GNN training and GCE process`')
    parser.add_argument("--val_ratio", default=0.2, type=float, help='val ratio for both GT-GNN training and GCE process')
    parser.add_argument("--splits", default='random', type=str, help="The split datasets way")
    parser.add_argument("--device", default=0, type=int, help='GPU device')
    parser.add_argument("--seed", default=0, type=int, help='random seed')
    parser.add_argument("-bsz", "--batch_size", default=32, type=int)
    parser.add_argument("-nat","--num_atom_types", default=len(NODE_LABEL_MAP['AIDS']), type=int, help='num atom types')
    parser.add_argument("-at","--ablation_type", default=None, choices=['np', 'nt', 'nf'],
                help='np - no pretrained text encoder, nt - no training of text encoder, nf - no feedback')
    parser.add_argument("-gt", "--generate_text", default=1, type=int, help='generate text attribute for the graph')
    
    # GNN args
    parser.add_argument("-ge", "--gnn_epochs", default=500, type=int, help='GNN model epochs')
    parser.add_argument("-glr", "--gnn_lr", default=1e-3, type=float, help='GNN model learning rate')
    parser.add_argument('-ged', "--gnn_embedding_dim", default=32, type=int, help='GNN model embedding dim')
    parser.add_argument("-gwd", "--gnn_weight_decay", default=0.001, type=float, help='GNN model weight decay')

    # Autoencoder args
    parser.add_argument("-lmm", "--lm_model", default="Bert", type=str, help='LM model used in the autoencoder')
    parser.add_argument("-ehd", "--exp_h_dim", default=512, type=int, help='autoencoder hidden dim')
    parser.add_argument("-mcl", "--max_context_length", default=4096, type=int, help='max context length')
    parser.add_argument("-emm", "--exp_m_mu", default=1.0, type=float, help='multiplied weight for the similarity loss')
    parser.add_argument("-ecm", "--exp_c_mu", default=0.5, type=float, help='multiplied weight for the prediction loss')
    parser.add_argument("-epe", "--exp_pretrain_epochs", default=50, type=int, help='pretrain epochs for text encoder')
    parser.add_argument("-eplr", "--exp_pretrain_lr", default=1e-6, type=float, help='pretrain lr for text encoder')
    parser.add_argument("-epwd", "--exp_pretrain_weight_decay", default=1e-5, type=float,help='pretrain weight decay for text encoder')
    parser.add_argument("-ete", "--exp_train_epochs", default=3, type=int, help='train restart rounds for autoencoder')
    parser.add_argument("-etlr", "--exp_train_lr", default=1e-6, type=float, help='train lr for autoencoder')
    parser.add_argument("-etwd", "--exp_train_weight_decay", default=1e-5, type=float, help='train weight decay for autoencoder')
    parser.add_argument("-eft", "--exp_feedback_times", default=3, type=int, help='LLM feedback times for autoencoder')
    parser.add_argument("-etpf", "--exp_train_steps_per_feedback", default=20, type=int, help='train steps between CTA feedback for autoencoder')
    parser.add_argument("-ed", "--exp_dropout", default=0.1, type=float, help='dropout for autoencoder')
    parser.add_argument("-et", "--exp_train", default=1, type=int, help='train autoencoder')
    parser.add_argument("-etp", "--exp_train_percent", default=0.2, type=float, help='percent of training data to use when training autoencoder')

    return parser.parse_args()


def llm_gce(args, dataset, gnn, exp_num):
    data_split = [0.5, 0.25, 0.25] 
    pretrain_train_loader, pretrain_val_loader, _ = dataset.get_dataloaders(args.batch_size, data_split, mask_pos = True)
    explainer_train_loader, explainer_val_loader, explainer_test_loader =  dataset.get_dataloaders(4, data_split, mask_pos = True)
     
    gnn.eval()

    # Load autoencoder
    explainer = GraphTextClipModel(
            text_encoder=AutoModel.from_pretrained(MODEL_PRETRAIN_MAP[args.lm_model]),
            tokenizer=AutoTokenizer.from_pretrained(MODEL_PRETRAIN_MAP[args.lm_model]),
            lmconfig=AutoConfig.from_pretrained(MODEL_PRETRAIN_MAP[args.lm_model]),
            graph_encoder=gnn,
            args=args,
            graph_emb_dim=args.gnn_embedding_dim,
            max_num_nodes=dataset.max_num_nodes,
        ).to(args.device)
    
    # pretrain
    if args.ablation_type != "np":
        pretrain_lm_path = './saved_models/lm_'+args.dataset +'.pth'
        if os.path.isfile(pretrain_lm_path):
            print('----------------------Loading Pretrained LM----------------------\n')
            explainer.load_state_dict(torch.load(pretrain_lm_path))
            print('----------------------Pretrained LM Loaded----------------------\n')
        else:
            print('----------------------Training Pretrained LM----------------------\n')
            explainer = pretrain_autoencoder(args, explainer, pretrain_train_loader, pretrain_val_loader)
            torch.save(explainer.state_dict(), pretrain_lm_path)
            print('----------------------Pretrained LM Saved----------------------\n')


    print('----------------------Training Autoencoder----------------------\n')
    if args.ablation_type != None:
        explainer_path = './saved_models/' + args.ablation_type + '_explainer_'+args.dataset+ '_exp_num'+str(exp_num)+'.pth'
    else:
        explainer_path = './saved_models/explainer_'+args.dataset+ '_exp_num'+str(exp_num)+'.pth'
    if args.exp_train == 0:
        explainer.load_state_dict(torch.load(explainer_path))
    else:
        explainer = train_autoencoder(args, explainer, explainer_train_loader, explainer_val_loader)
    if args.ablation_type != "nt":
        torch.save(explainer.state_dict(), explainer_path)
    print('----------------------Autoencoder Trained and Saved----------------------\n')

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
    return explainer, explainer_test_loader

def main():
    args = parse_llm_gce_args()
    args.device='cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu'
    args.num_atom_types = len(NODE_LABEL_MAP[args.dataset]) + 1

    set_seed()
    dataset = Dataset(dataset=args.dataset, generate_text=args.generate_text)
    gnn = gnn_trainer(args, dataset) 
 
    # Define the file paths
    save_file_path = f'./exp_results/ablation/{args.ablation_type}_{args.dataset}.csv' if args.ablation_type is not None else f'./exp_results/{args.dataset}.csv'
    # proximity_file_path = f'./exp_results/ablation/{args.ablation_type}_{args.dataset}_proximity.csv' if args.ablation_type is not None else f'./exp_results/{args.dataset}_proximity.csv'

    start = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    # Mark the beginning of the experiments
    write_file_start(start, save_file_path)

    validity_list, proximity_list, validity_without_chem_list, proximity_without_chem_list = [], [], [], []
    for exp_num in range(args.num_exps):
        explainer, explainer_test_loader = llm_gce(args, dataset, gnn, exp_num)
        validity, proximity, validity_without_chem, proximity_without_chem = evaluate_gce_model(explainer_test_loader, gnn, dataset, explainer) 
        validity_list.append(validity)
        proximity_list.append(proximity)
        validity_without_chem_list.append(validity_without_chem)
        proximity_without_chem_list.append(proximity_without_chem)
        
        # Write the results immediately to the files
        with open(save_file_path, "a") as f:
            f.write(f'\nExperiment {exp_num}, {validity}, {proximity}, {validity_without_chem}, {proximity_without_chem}')


    # write the mean and standart deviation to the validity and proximity files
    with open(save_file_path, "a") as f:
        # f_validity.write(f'\n{np.mean(validity_list)} ± {np.std(validity_list)}')
        # f_proximity.write(f'\n{np.mean(proximity_list)} ± {np.std(proximity_list)}')
        f.write(f'\nSummary,{np.mean(validity_list)} ± {np.std(validity_list)}, \
                {np.mean(proximity_list)} ± {np.std(proximity_list)}, \
                {np.mean(validity_without_chem_list)} ± {np.std(validity_without_chem_list)}, \
                {np.mean(proximity_without_chem_list)} ± {np.std(proximity_without_chem_list)}\nEnd Experiment') \
        

    print(f'validity: {np.mean(validity_list)} ± {np.std(validity_list)}')
    print(f'proximity: {np.mean(proximity_list)} ± {np.std(proximity_list)}\nEnd Experiment')

    print('----------------------Experiment Done----------------------\n')
                
if '__main__' == __name__:
    main()