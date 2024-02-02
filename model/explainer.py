import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput
from typing import Optional, Any, Tuple, List
from utils.smiles import *

@dataclass
class DualModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    adj_reconst: torch.FloatTensor = None
    SMILES: List[str] = None
    true_prob: torch.FloatTensor = None
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "gnn_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )

class GraphTextClipModel(PreTrainedModel):
    def __init__(self, text_encoder, graph_encoder, lmconfig, args, max_num_nodes=100, 
                 emb_dim=768, graph_emb_dim=16, tokenizer=None):
        '''
        @text_encoder: Bert, etc.
        @graph_encoder: GCN, etc.
        '''
        super().__init__(lmconfig) 
        self.max_num_nodes = max_num_nodes
        self.num_atom_types = args.num_atom_types
        self.dropout_value = args.exp_dropout
        self.h_dim = args.exp_h_dim
        self.max_context_length = args.max_context_length
        self.m_mu = args.exp_m_mu
        self.c_mu = args.exp_c_mu

        self.graph_encoder = graph_encoder.to(args.device)
        self.text_encoder = text_encoder.to(args.device)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'right'

        self.dropout = nn.Dropout(self.dropout_value)
        self.logit_scale = nn.Parameter(torch.ones([]) * 1).to(args.device)
        self.transform = nn.Linear(emb_dim, graph_emb_dim)

        self._device = args.device
         
        self.decoder_adj = nn.Sequential(
            nn.Linear(args.gnn_embedding_dim, self.h_dim), 
            nn.BatchNorm1d(self.h_dim), 
            nn.Dropout(self.dropout_value), 
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim), 
            nn.BatchNorm1d(self.h_dim), 
            nn.Dropout(self.dropout_value), 
            nn.ReLU(),
            nn.Linear(self.h_dim, self.max_num_nodes*self.max_num_nodes), 
            nn.Sigmoid()
        ).to(args.device)

        self.decoder_edge = nn.Sequential(
            nn.Linear(args.gnn_embedding_dim, self.h_dim), 
            nn.BatchNorm1d(self.h_dim), 
            nn.Dropout(self.dropout_value), 
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim), 
            nn.BatchNorm1d(self.h_dim), 
            nn.Dropout(self.dropout_value), 
            nn.ReLU(),
            nn.Linear(self.h_dim, self.max_num_nodes*self.max_num_nodes), 
            nn.Hardtanh(min_val=0, max_val=3.9) # 4->3.9: avoid one_hot index out of range
        ).to(args.device)
        
        self.decoder_x = nn.Sequential(
            nn.Linear(args.gnn_embedding_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout_value),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout_value),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.max_num_nodes*self.num_atom_types),
        ).to(args.device)
            
        '''
        batch.edge_index: shape: [batch_size, num_nodes, num_nodes]
        batch.mask: shape: [batch_size, num_nodes, num_nodes]
        '''
        self.graph_norm = nn.BatchNorm1d(self.h_dim)
        
        self.init_model()
        '''
        texts = [str]
        texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', padding_side='left')
        generation_outputs = padding, input_text, output_texts
        '''

    def init_model(self):
        for layer in self.decoder_adj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)
                nn.init.zeros_(layer.bias.data)

        nn.init.xavier_normal_(self.transform.weight.data)
        nn.init.zeros_(self.transform.bias.data)
        
        for param in self.graph_encoder.parameters():
            param.requires_grad = False
    
    def generate_smiles(self, adj_reconst: torch.Tensor, edge_reconst: torch.Tensor, x_reconst: torch.Tensor, dataset:str) -> List[str]:
        smiles = []
        for x, adj_matrix, edge_matrix in zip(x_reconst, adj_reconst, edge_reconst):
            mask = torch.BoolTensor([True] * x.size()[0])
            adj_matrix = adj_matrix * (1 - torch.eye(adj_matrix.shape[0])).to(adj_matrix.device)
            smile_str = graph_to_smiles(x, adj_matrix, edge_matrix, mask, dataset)
            
            smile_str = smile_str.split('.')
            smile_str = [substr for substr in smile_str if substr.replace("[", "").replace("]", "") not in NODE_LABEL_MAP[dataset].values()]
            smile_str = ".".join(smile_str)
            smiles.append(smile_str)
        return smiles
        
    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        graph_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + graph_loss) / 2.0
   
    def forward_clip(self, batch, return_loss=False):
        text = self.tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt').to(self.device)
        text_emb = self.text_encoder(**text)
        text_emb = text_emb.last_hidden_state[:, 0, :]
        text_emb = self.dropout(text_emb) 
        text_emb = self.transform(text_emb)       
        graph_emb = self.graph_encoder(batch)['rep_graph']
        # norm
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        graph_emb = graph_emb / graph_emb.norm(dim=-1, keepdim=True)
        # logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = text_emb @ graph_emb.T * logit_scale
        if return_loss:
            loss = self.clip_loss(logits_per_text)
        else:
            loss = torch.tensor(0.0)
        return {
            'loss': loss,
            'logits_per_text': logits_per_text,
            'text_embeds': text_emb,
            'graph_emb': graph_emb,
        }
    
    def CF_loss(self, adj_reconst, edge_reconst, x_reconst, batch):
        x, adj, edge_attr = batch['x'].to(self.device), batch['adj'].to(self.device), batch['edge_attr'].to(self.device)
        
        one_hot_x = F.one_hot(x.long().squeeze(-1), num_classes=self.num_atom_types).float()
        one_hot_x = one_hot_x.reshape(-1, self.max_num_nodes, self.num_atom_types)
        similarity_loss = -F.cosine_similarity(adj_reconst * edge_reconst, adj * edge_attr).abs().mean()\
                          -F.cosine_similarity(one_hot_x, x_reconst).abs().mean()
        
        desired_label = (1 - batch['label'].float()).long().to(self.device)
        new_mask = torch.ones(len(batch['x']), self.max_num_nodes, dtype=torch.bool, requires_grad=False)
        new_batch = {'x': x_reconst, 'adj': adj_reconst, 'edge_attr': edge_reconst, 'mask': new_mask}
        
        ground_truth_prediction = torch.exp(self.graph_encoder(new_batch, from_onehot=True)['y_pred'])
        conterfactual_loss = F.cross_entropy(ground_truth_prediction, desired_label)

        return self.m_mu * similarity_loss + self.c_mu * conterfactual_loss, ground_truth_prediction[:, 1]
    
    def forward_CF(self, batch, CF_text, return_loss=False):
        CF_text = self.tokenizer(CF_text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        CF_emb = self.text_encoder(**CF_text).last_hidden_state[:, 0, :]
        CF_emb = self.graph_encoder(batch)['rep_graph'] + self.transform(CF_emb)
        CF_emb = CF_emb / CF_emb.norm(dim=-1, keepdim=True)
        
        adj_reconst = self.decoder_adj(CF_emb)
        adj_reconst = adj_reconst.reshape(-1, self.max_num_nodes, self.max_num_nodes)
        adj_reconst = (adj_reconst + adj_reconst.transpose(1, 2)) / 2
        
        edge_reconst = self.decoder_edge(CF_emb)
        edge_reconst = edge_reconst.reshape(-1, self.max_num_nodes, self.max_num_nodes)
        edge_reconst = (edge_reconst + edge_reconst.transpose(1, 2)) / 2
        
        x_reconst = self.decoder_x(CF_emb)
        x_reconst = x_reconst.reshape(-1, self.max_num_nodes, self.num_atom_types)
        x_reconst = F.softmax(x_reconst, dim=-1)
        
        if return_loss:
            loss, true_prob = self.CF_loss(
                adj_reconst=adj_reconst, edge_reconst=edge_reconst, x_reconst=x_reconst, batch=batch
            )
        else: 
            loss = torch.tensor(0.0)

        SMILES = self.generate_smiles(adj_reconst=adj_reconst, edge_reconst=edge_reconst, 
                                      x_reconst=torch.argmax(x_reconst, dim=-1, keepdim=True), dataset=batch['dataset'][0])
        
        return DualModelOutput(loss=loss, adj_reconst=adj_reconst, SMILES=SMILES, 
                               true_prob=true_prob.detach().cpu().numpy().tolist())