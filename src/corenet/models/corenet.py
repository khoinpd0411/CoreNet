import torch
import torch.nn as nn

from corenet.layers import FirstAwareBranch, SecAwareBranch, MLP

from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input, SparseFeat, VarLenSparseFeat
from deepctr_torch.layers import FM, concat_fun

class CoreNet(BaseModel):
    """Instantiates the CoreNet Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of dnn
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given MLP coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_ln: bool. Whether use LayerNormalization before activation or not in MLP
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on ``device`` . ``gpus[0]`` should be the same gpu with ``device`` .
    :return: A PyTorch model instance.

    """

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns,
                 dnn_hidden_units=(256,),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0.1,
                 dnn_activation='relu', use_ln=True, task='binary', device='cpu', gpus=None):
        super(CoreNet, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                   l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                   device=device, gpus=gpus)

        if not len(dnn_hidden_units) > 0:
            raise ValueError("dnn_hidden_units is null!")    
            
        self.fm = FM()

        self.sec_integrate = SecAwareBranch()
        
        self.sparse_feat_num = len(list(filter(lambda x: isinstance(x, SparseFeat) or isinstance(x, VarLenSparseFeat),
                                               dnn_feature_columns)))

        self.first_integrate = FirstAwareBranch(self.sparse_feat_num, self.embedding_size, device = device)

        self.first_aware = MLP(self.embedding_size * self.sparse_feat_num,
                                dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn,
                                dropout_rate=dnn_dropout,
                                use_ln=use_ln, init_std=init_std, device=device)
        
        self.sec_aware = MLP(self.embedding_size * self.sparse_feat_num,
                                dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn,
                                dropout_rate=dnn_dropout,
                                use_ln=use_ln, init_std=init_std, device=device)
        
        self.first_reweight = nn.Linear(
            dnn_hidden_units[-1], self.sparse_feat_num, bias=False).to(device)
        self.sec_reweight = nn.Linear(
            dnn_hidden_units[-1], self.sparse_feat_num, bias=False).to(device)
        
        self.ln = nn.LayerNorm(self.embedding_size * self.sparse_feat_num)

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.first_aware.named_parameters()),
            l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.sec_aware.named_parameters()),
            l2=l2_reg_dnn)
        
        self.add_regularization_weight(self.sec_reweight.weight, l2=l2_reg_dnn)
        self.add_regularization_weight(self.first_reweight.weight, l2=l2_reg_dnn)

        self.to(device)

    def forward(self, X):
        sparse_embedding_list, _ = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                   self.embedding_dict)
        if not len(sparse_embedding_list) > 0:
            raise ValueError("there are no sparse features")

        fm_input = concat_fun(sparse_embedding_list, axis=1)

        sec_out = self.sec_integrate(fm_input)
        sec_out = self.ln(sec_out)
        sec_out = self.sec_aware(sec_out)
        m_sec = self.sec_reweight(sec_out)

        first_out = self.first_integrate(fm_input)
        first_out = self.ln(first_out)
        first_out = self.first_aware(first_out)
        m_first= self.first_reweight(first_out)

        m_final = m_first + m_sec

        logit = self.linear_model(X, sparse_feat_refine_weight=m_final)

        fm_input = torch.cat(sparse_embedding_list, dim=1)
        refined_fm_input = fm_input * m_final.unsqueeze(-1)  # \textbf{v}_{x,i}=m_{x,i} * \textbf{v}_i
        fm_output = self.fm(refined_fm_input)
        logit += fm_output

        y_pred = self.out(logit)

        return y_pred