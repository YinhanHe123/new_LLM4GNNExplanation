import torch
import torch.nn as nn
import torch.nn.functional as F

# class DenseGATConv(nn.Module):
#     def __init__(self, in_channels, out_channels, edge_attr_dim, dataset='synthetic', aggr='add', bias=True):
#         super(DenseGATConv, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.edge_attr_dim = edge_attr_dim
#         self.aggr = aggr
#         # Linear transformations for node features
#         self.lin_rel = nn.Linear(in_channels, out_channels, bias=bias)
#         self.lin_root = nn.Linear(in_channels, out_channels, bias=False)

#         # Additional transformation for edge attributes
#         self.edge_embedding = nn.Embedding(edge_attr_dim + 1, 1) # consider 0 for no edge

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin_rel.reset_parameters()
#         self.lin_root.reset_parameters()
#         self.edge_embedding.reset_parameters()

#     def forward(self, x, adj, edge_attr, mask=None):
#         '''
#         @edge_attr: [batch_size, max_num_nodes, max_num_nodes], element in {0, 1, 2, 3}
#         @adj: [batch_size, max_num_nodes, max_num_nodes], element in {0, 1}
#         '''
#         B, N, _ = x.size()

#         # Transform edge attributes
#         edge_attr_transformed = self.edge_embedding(
#             edge_attr.long().view(-1)
#         ).view(B, N, N)  # Shape: [B, N, N]
#         adj = adj * edge_attr_transformed  # Shape: [B, N, N]

#         # Perform graph convolution
#         out = torch.matmul(adj, x)  # Shape: [B, N, out_channels]

#         if self.aggr == 'mean':
#             out = out / adj.sum(dim=-1, keepdim=True).clamp_(min=1)

#         out = self.lin_rel(out)
#         out += self.lin_root(x)

#         if mask is not None:
#             out = out * mask.view(B, N, 1).to(x.dtype)

#         return out

#     def __repr__(self):
#         return (f'{self.__class__.__name__}({self.in_channels}, '
#                 f'{self.out_channels}, edge_attr_dim={self.edge_attr_dim})')

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

    def forward(self, x, adj, edge_attr, mask=None, from_onehot=True):
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
    def __init__(self, num_features, num_classes, embedding_dim, device):
        super(GCN, self).__init__()
        self.num_features = num_features
        self.embedding = nn.Linear(num_features, embedding_dim)
        self.conv1 = DenseGATConv(embedding_dim, 32, 4)
        self.conv2 = DenseGATConv(32, 16, 4)
        self.fc = nn.Linear(16, num_classes) 
        self.device = device

    def forward(self, data, use_softmax=True, from_onehot=False):
        x, adj, mask, edge_attr = data['x'].to(self.device), data['adj'].to(self.device), \
                                  data['mask'].to(self.device), data['edge_attr'].to(self.device)
        if not from_onehot:
            x = F.one_hot(x.long().squeeze(-1), num_classes=self.num_features).float()
        
        x = self.embedding(x.float())
        x = F.relu(self.conv1(x, adj, edge_attr, mask=mask))
        x = F.relu(self.conv2(x, adj, edge_attr, mask=mask))

        mask_count = mask.sum(dim=1, keepdim=True)
        mask = mask.unsqueeze(-1)
        x = x * mask
        
        x = x.sum(dim=1) / mask_count
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc(x)
        if use_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x
    
    def forward_emb(self, data, from_onehot=False):
        x, adj, mask, edge_attr = data['x'].to(self.device), data['adj'].to(self.device), \
                                  data['mask'].to(self.device), data['edge_attr'].to(self.device)
        if not from_onehot:
            x = F.one_hot(x.long().squeeze(-1), num_classes=self.num_features).float()
        
        x = self.embedding(x.float())
        x = F.relu(self.conv1(x, adj, edge_attr, mask=mask))
        x = F.relu(self.conv2(x, adj, edge_attr, mask=mask))
        
        mask_count = mask.sum(dim=1, keepdim=True)
        mask = mask.unsqueeze(-1)
        x = x * mask
        
        x = x.sum(dim=1) / mask_count  
        return x
    
class Graph_pred_model(nn.Module):
    def __init__(self, x_dim, h_dim, n_out, num_edge_attr, max_num_nodes, device):
        super(Graph_pred_model, self).__init__()
        self.num_graph_models = 3
        self.device = device
        self.graph_model = nn.ModuleList([DenseGATConv(x_dim, h_dim, num_edge_attr).to(device) for i in range(self.num_graph_models)])
        self.encoder = nn.Sequential(nn.Linear(2 * h_dim, h_dim), nn.ReLU())
        self.predictor = nn.Sequential(nn.Linear(h_dim, n_out))
        self.max_num_nodes = max_num_nodes
        self.mask = torch.nn.Parameter(torch.ones(max_num_nodes), requires_grad=True)
        self.register_parameter("mask", self.mask)
        self.nuM_edge_attr = num_edge_attr

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

    def forward(self, data):
        x, adj, mask, edge_attr = data['x'].to(self.device), data['adj'].to(self.device), \
                                  data['mask'].to(self.device), data['edge_attr'].to(self.device)
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