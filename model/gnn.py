import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm

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

def train_gnn(args, gnn, gnn_train_loader, gnn_val_loader):
    print("Training GNN")
    optimizer = torch.optim.Adam(gnn.parameters(), lr=args.gnn_lr, weight_decay=args.gnn_weight_decay)
    gnn.train()
    gnn_start_train_time = time.time()
    best_eval_acc = 0

    for epoch in tqdm(range(args.gnn_epochs)):
        train_loss = 0
        train_acc = 0
        for batch in gnn_train_loader:
            labels = batch['label'].to(args.device)
            optimizer.zero_grad()
            out = gnn(batch)
            out = out['y_pred']
            loss = F.nll_loss(out,labels.long())
            loss.backward()
            optimizer.step()
            
            train_loss += loss
            train_acc += (out.argmax(dim=1) == labels).sum().item()
        
        if epoch % 10 == 0 or epoch == args.gnn_epochs - 1:
            with torch.no_grad():
                train_loss /= len(gnn_train_loader)
                train_acc /= len(gnn_train_loader.sampler)
                
                eval_loss = 0
                eval_acc = 0
                for batch in gnn_val_loader:
                    labels = batch['label'].to(args.device)
                    out = gnn(batch)
                    out = out['y_pred']
                    eval_loss += F.nll_loss(out,labels.long()).item()
                    eval_acc += (out.argmax(dim=1) == labels).sum().item()
                eval_loss /= len(gnn_val_loader)
                eval_acc /= len(gnn_val_loader.sampler)
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    best_model = gnn.state_dict()    
            time_checkpoint = time.time()
            time_comsumed = time_checkpoint - gnn_start_train_time
            print(f'epoch: {epoch} | train_loss: {train_loss} | train_acc : {train_acc} | eval_loss: {eval_loss} | eval_acc: {eval_acc} | time_consumed: {time_comsumed}')
    gnn.load_state_dict(best_model)
    return gnn


def test_gnn(gnn, gnn_test_loader):
    print("Testing GNN")
    test_loss = 0
    test_acc = 0
    for batch in gnn_test_loader:
        labels = batch['label'].to(gnn.device)
        out = gnn(batch)
        out = out['y_pred']
        test_loss += F.nll_loss(out,labels.long()).item()
        test_acc += (out.argmax(dim=1) == labels).sum().item()
    test_loss /= len(gnn_test_loader)
    test_acc /= len(gnn_test_loader.sampler)
    print(f'test_loss: {test_loss} | test_acc : {test_acc}')


def gnn_trainer(args, dataset):
    data_split = [0.5, 0.25, 0.25] 
    gnn_train_loader, gnn_val_loader, gnn_test_loader = dataset.get_dataloaders(args.batch_size, data_split)
    gnn = GCN(num_features=args.num_atom_types, embedding_dim=args.gnn_embedding_dim, num_classes=2, num_edge_attr=4, device=args.device)
    gnn_path = './saved_models/gnn_'+args.dataset+'.pth'
    if os.path.isfile(gnn_path):
        print('----------------------Loading GT-GNN----------------------\n')
        gnn.load_state_dict(torch.load(gnn_path))
        print('----------------------GT-GNN Loaded----------------------\n')
    else:
        print('----------------------Training GT-GNN----------------------\n')
        gnn = train_gnn(args, gnn, gnn_train_loader, gnn_val_loader)
        test_gnn(gnn, gnn_test_loader)
        torch.save(gnn.state_dict(), gnn_path)
        print('gnn model saved to {}'.format(gnn_path))
        print('----------------------GT-GNN Saved----------------------\n')
    return gnn