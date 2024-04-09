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
import time

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
    parser.add_argument("-encoder", "--encoder_model", default="Bert", type=str, help='LM model used in the autoencoder')
    parser.add_argument("-llm", "--llm_model", 
                        choices=[
                            "gpt-3.5-turbo-1106",
                            "ncfrey/ChemGPT-1.2B",
                            "prajwalsahu5/GPT-Molecule-Generation",
                            "gpt-4-0125-preview",
                        ],
                        default="gpt-3.5-turbo-1106", 
                        type=str, 
                        help='LLM model used in the feedback')
    parser.add_argument("-ehd", "--exp_h_dim", default=512, type=int, help='autoencoder hidden dim')
    parser.add_argument("-mcl", "--max_context_length", default=4096, type=int, help='max context length')
    parser.add_argument("-emm", "--exp_m_mu", default=1.0, type=float, help='multiplied weight for the similarity loss')
    parser.add_argument("-ecm", "--exp_c_mu", default=0.5, type=float, help='multiplied weight for the prediction loss')
    parser.add_argument("-epe", "--exp_pretrain_epochs", default=100, type=int, help='pretrain epochs for text encoder')
    parser.add_argument("-eplr", "--exp_pretrain_lr", default=1e-2, type=float, help='pretrain lr for text encoder')
    parser.add_argument("-epwd", "--exp_pretrain_weight_decay", default=1e-6, type=float,help='pretrain weight decay for text encoder')
    parser.add_argument("-ete", "--exp_train_epochs", default=3, type=int, help='train restart rounds for autoencoder')
    parser.add_argument("-etlr", "--exp_train_lr", default=1e-2, type=float, help='train lr for autoencoder')
    parser.add_argument("-etwd", "--exp_train_weight_decay", default=1e-5, type=float, help='train weight decay for autoencoder')
    parser.add_argument("-eft", "--exp_feedback_times", default=3, type=int, help='LLM feedback times for autoencoder')
    parser.add_argument("-etpf", "--exp_train_steps_per_feedback", default=20, type=int, help='train steps between CTA feedback for autoencoder')
    parser.add_argument("-ed", "--exp_dropout", default=0.1, type=float, help='dropout for autoencoder')
    parser.add_argument("-et", "--exp_train", default=1, type=int, help='train autoencoder')
    parser.add_argument("-etp", "--exp_data_percent", default=0.2, type=float, help='percent of data to use when training / testing autoencoder')

    return parser.parse_args()


def llm_gce(args, dataset, gnn, exp_num):
    data_split = [0.5, 0.25, 0.25] 
    pretrain_train_loader, pretrain_val_loader, _ = dataset.get_dataloaders(args.batch_size, data_split, mask_pos = True)
    explainer_train_loader, explainer_val_loader, explainer_test_loader =  dataset.get_dataloaders(4, data_split, mask_pos = True)
     
    gnn.eval()

    # Load autoencoder
    explainer = GraphTextClipModel(
            text_encoder=AutoModel.from_pretrained(MODEL_PRETRAIN_MAP[args.encoder_model]),
            tokenizer=AutoTokenizer.from_pretrained(MODEL_PRETRAIN_MAP[args.encoder_model]),
            lmconfig=AutoConfig.from_pretrained(MODEL_PRETRAIN_MAP[args.encoder_model]),
            graph_encoder=gnn,
            args=args,
            graph_emb_dim=args.gnn_embedding_dim,
            max_num_nodes=dataset.max_num_nodes,
            llm_model=args.llm_model,
        ).to(args.device)
    
    # pretrain
    if args.ablation_type != "np":
        if args.encoder_model == 'Bert':
            pretrain_lm_path = './saved_models/lm_'+args.dataset + '_eplr_' + str(args.exp_pretrain_lr) + '_eplw' + str(args.exp_pretrain_weight_decay) + '_epe_'+ str(args.exp_pretrain_epochs) + '.pth'
        else:
            pretrain_lm_path = './saved_models/'+args.encoder_model+'_'+args.dataset + '_eplr_' + str(args.exp_pretrain_lr) + '_eplw' + str(args.exp_pretrain_weight_decay) + '_epe_'+ str(args.exp_pretrain_epochs) + '_'+ args.encoder_model + '.pth'
        if os.path.isfile(pretrain_lm_path):
            print('----------------------Loading Pretrained LM----------------------\n')
            pretrain_state_dict = torch.load(pretrain_lm_path, map_location=args.device)
            explainer.load_state_dict(pretrain_state_dict)
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
        explainer_path = './saved_models/explainer_'+args.dataset+ '_etlr_' + str(args.exp_train_lr) + '_etlw' + str(args.exp_train_weight_decay) + "_feedback_" + str(args.exp_feedback_times) + '_exp_num'+str(exp_num)+'.pth'
    if args.exp_train == 0:
        state_dict = torch.load(explainer_path, map_location=args.device)
        explainer.load_state_dict(state_dict)
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
    args_list = (
         f'{args.dataset}_'
         f'pretrain_lr_{args.exp_pretrain_lr}_'
         f'pretrained_epochs_{args.exp_pretrain_epochs}_'
         f'pretrained_weight_decay_{args.exp_pretrain_weight_decay}_'
         f'm_mu_{args.m_mu}'
         f'c_mu_{args.c_mu}'
         f'train_lr_{args.exp_train_lr}_'
         f'train_epochs_{args.exp_train_epochs}_'
         f'train_weight_decay_{args.exp_train_weight_decay}_'
         f'feedback_times_{args.exp_feedback_times}_'
         f'train_steps_per_feedback_{args.exp_train_steps_per_feedback}_'
         f'train_data_percent_{args.exp_data_percent}.csv'
         f'encoder_{args.encoder_model}_',
         f'llm_{args.llm_model}_',
         )
    save_file_path = f'./exp_results/{args_list}.csv' if args.ablation_type == None else f'./exp_results/ablation/{args.ablation_type}_{args_list}.csv'
    
    start = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    # Mark the beginning of the experiments
    write_file_start(start, save_file_path)

    validity_list, proximity_list, validity_without_chem_list, proximity_without_chem_list = [], [], [], []
    total_time = 0
    for exp_num in range(args.num_exps):
        begin = time.time()
        explainer, explainer_test_loader = llm_gce(args, dataset, gnn, exp_num)
        validity, proximity, validity_without_chem, proximity_without_chem, cf_results = evaluate_gce_model(explainer_test_loader, gnn, dataset, explainer) 
        
        cf_results.to_csv(open(f"./exp_results/{args_list}_exp{exp_num}_cfs.csv", "w"), index=False)
        validity_list.append(validity)
        proximity_list.append(proximity)
        validity_without_chem_list.append(validity_without_chem)
        proximity_without_chem_list.append(proximity_without_chem)
        end = time.time()
        timing = end - begin
        total_time += timing
        # Write the results immediately to the files
        with open(save_file_path, "a") as f:
            f.write(f'\nExperiment {exp_num}, {validity}, {proximity}, {validity_without_chem}, {proximity_without_chem}, {timing} seconds')


    # write the mean and standart deviation to the validity and proximity files
    with open(save_file_path, "a") as f:
        f.write(f'\nSummary,{np.mean(validity_list)} ± {np.std(validity_list)}, \
                {np.mean(proximity_list)} ± {np.std(proximity_list)}, \
                {np.mean(validity_without_chem_list)} ± {np.std(validity_without_chem_list)}, \
                {np.mean(proximity_without_chem_list)} ± {np.std(proximity_without_chem_list)}\nEnd Experiment, total time {total_time}.') \
        

    print(f'validity: {np.mean(validity_list)} ± {np.std(validity_list)}')
    print(f'proximity: {np.mean(proximity_list)} ± {np.std(proximity_list)}\nEnd Experiment')

    print('----------------------Experiment Done----------------------\n')
                
if '__main__' == __name__:
    main()