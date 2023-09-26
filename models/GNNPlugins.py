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
    
    self.conv_layers = torch.nn.ModuleList([])
    self.transf_layers = torch.nn.ModuleList([])
    self.bn_layers = torch.nn.ModuleList([])

    self.conv1 = torch_geometric.nn.TransformerConv(self.feature_size, 
                                self.embedding_size, 
                                heads=self.n_heads, 
                                dropout=dropout_rate,
                                edge_dim=self.edge_dim,
                                beta=beta)
    self.transf1 = torch.nn.Linear(self.embedding_size*self.n_heads, self.embedding_size)
    self.bn1 = torch.nn.BatchNorm1d(self.embedding_size) 

    for i in range(self.n_layers):
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
    x = self.conv1(x, edge_index, edge_attr)
    x = torch.nn.functional.relu(self.transf1(x))
    x = self.bn1(x)

    for i in range(self.n_layers):
      x = self.conv_layers[i](x, edge_index, edge_attr)
      x = torch.nn.functional.relu(self.transf_layers[i](x))
      x = self.bn_layers[i](x)

    x = self.out(x)
    return x
