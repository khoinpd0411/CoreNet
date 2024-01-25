import torch
import torch.nn as nn
import torch.nn.functional as F

from deepctr_torch.layers.activation import activation_layer

class FirstAwareBranch(nn.Module):
    """
    First order-aware Component

    Arguments
    - **sparse_feat_num**: number of feature.
    - **embedding_size**: embedding size.
    - **seed**: A Python integer to use as random seed.
    """
    def __init__(self, sparse_feat_num, embedding_size, seed=1024, device='cpu'):
        super(FirstAwareBranch, self).__init__()
        self.W = nn.Parameter(torch.Tensor(sparse_feat_num, embedding_size))
        self.seed = seed

        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

        self.to(device)

    def forward(self, inputs):
        outputs = torch.mul(inputs, self.W)
        outputs = outputs.reshape(outputs.shape[0], -1)
        return outputs
    
class SecAwareBranch(nn.Module):
    """
    Second order-aware Component
    """

    def __init__(self):
        super(SecAwareBranch, self).__init__()

    def forward(self, inputs):
        field_size = inputs.shape[1]

        fm_input = inputs.unsqueeze(1).repeat(1, field_size, 1, 1)
        square = torch.pow(inputs, 2)
        
        cross_term = torch.sum(torch.mul(fm_input, inputs.unsqueeze(2)), dim = 2)
        cross_term = cross_term - square
        cross_term = cross_term.reshape(cross_term.shape[0], -1)
        return cross_term

class MLP(nn.Module):
    
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **inputs_dim**: input feature dimension.
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_ln**: bool. Whether use LayerNormalization before activation or not.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_ln=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(MLP, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_ln = use_ln

        if use_ln:
            self.ln = nn.ModuleList(
                    [nn.LayerNorm(hidden_units[i]) for i in range(len(hidden_units))])

        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])


        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)

            if self.use_ln:
                fc = self.ln[i](fc)

            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)

            deep_input = fc
        return deep_input