import torch
import torch_geometric

class GNNModel(torch.nn.Module):
  def __init__(self, embedding_size, n_layers, feature_size, n_heads, dropout_rate, edge_dim, num_classes, beta=True):
    super(GNNModel, self).__init__()
    self.n_layers = n_layers
    self.embedding_size = embedding_size
    self.feature_size = feature_size
    self.n_heads = n_heads
    self.edge_dim = edge_dim
    self.num_classes = num_classes

    self.skip_layers = torch.nn.ModuleList([])
    self.conv_layers = torch.nn.ModuleList([])
    self.transf_layers = torch.nn.ModuleList([])
    self.bn_layers = torch.nn.ModuleList([])

    self.lin_skip1 = torch.nn.Linear(self.feature_size, self.embedding_size * self.n_heads)
    self.conv1 = torch_geometric.nn.TransformerConv(self.feature_size,   # You can change other GNN models from here such as
                                self.embedding_size,                     # GCN, GAT, GraphSAGE, etc,...
                                heads=self.n_heads, 
                                dropout=dropout_rate,
                                edge_dim=self.edge_dim,
                                beta=beta)
    self.transf1 = torch.nn.Linear(self.embedding_size*self.n_heads, self.embedding_size)
    self.bn1 = torch.nn.BatchNorm1d(self.embedding_size) 

    for i in range(self.n_layers):
      self.skip_layers.append(torch.nn.Linear(self.embedding_size, self.embedding_size * self.n_heads))
      self.conv_layers.append(torch_geometric.nn.TransformerConv(self.embedding_size, 
                                              self.embedding_size, 
                                              heads=self.n_heads, 
                                              dropout=dropout_rate,
                                              edge_dim=self.edge_dim,
                                              beta=beta))

      self.transf_layers.append(torch.nn.Linear(self.embedding_size*self.n_heads, self.embedding_size))
      self.bn_layers.append(torch.nn.BatchNorm1d(self.embedding_size))
    
    self.out = torch.nn.Linear(self.embedding_size, self.num_classes)

  def forward(self, x, edge_index, edge_attr):
    x_skip = self.lin_skip1(x)
    x = self.conv1(x, edge_index, edge_attr)
    x += x_skip
    x = torch.nn.functional.relu(self.transf1(x))
    x = self.bn1(x)

    for i in range(self.n_layers):
      x_skip = self.skip_layers[i](x)
      x = self.conv_layers[i](x, edge_index, edge_attr)
      x += x_skip
      x = torch.nn.functional.relu(self.transf_layers[i](x))
      x = self.bn_layers[i](x)

    x = self.out(x)
    return x

class AttentionModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes, aggr='mean'):
        super(AttentionModel, self).__init__()
        self.input_dim = input_dim
        self.aggr = aggr
        self.lin_skip = torch.nn.Linear(input_dim, input_dim)
        self.query = torch.nn.Linear(input_dim, input_dim)
        self.key = torch.nn.Linear(input_dim, input_dim)
        self.value = torch.nn.Linear(input_dim, input_dim)
        self.softmax = torch.nn.Softmax(dim=1)
        self.fc = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x_r = self.lin_skip(x)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.mm(queries, keys.transpose(0, 1)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.mm(attention, values)
        if self.aggr == 'mean': weighted /= x.shape[0]
        weighted += x_r
        return self.fc(weighted)


class ImprovedAttentionModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ImprovedAttentionModel, self).__init__()
        self.input_dim = input_dim
        self.lin_skip = torch.nn.Linear(input_dim, input_dim)
        self.lin_beta = torch.nn.Linear(3 * input_dim, 1)
        self.query = torch.nn.Linear(input_dim, input_dim)
        self.key = torch.nn.Linear(input_dim, input_dim)
        self.value = torch.nn.Linear(input_dim, input_dim)
        self.softmax = torch.nn.Softmax(dim=1)
        self.fc = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x_r = self.lin_skip(x)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.mm(queries, keys.transpose(0, 1)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        x = torch.mm(attention, values)
        beta = self.lin_beta(torch.cat([x, x_r, x - x_r], dim=-1))
        beta = beta.sigmoid()
        x = beta * x_r + (1 - beta) * x
        return self.fc(x)
