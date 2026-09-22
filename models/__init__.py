from typing import Union

import torch

from .hkst.model import hkst
from .koopa.koopa import koopa
from .dlinear.DLinear import dlinear
from .patchmlp.PatchMLP import patchmlp
from .patchtst.PatchTST import patchtst
from .itransformer.iTransformer import itransformer
from .stwave.models import stwave
from .stdn.model import stdn
from .staeformer.model import staeformer
from .pdformer.model import pdformer
# from .dstmamba.DSTMamba_Arch import dstmamba


def get_model(model_type: str, **kwargs: Union[int, float]) -> torch.nn.Module:
    if model_type == 'kwast':
        from .kwast.models import kwast
        model = kwast(kwargs['num_nodes'], kwargs['input_len'], kwargs['output_len'],
                      kwargs['train_loader'], kwargs['gso'], kwargs['device'])

    elif model_type == 'koopa':
        model = koopa(kwargs['num_nodes'], kwargs['input_len'], kwargs['output_len'], kwargs['train_loader'])
    elif model_type == 'dlinear' or model_type == 'dlinear_ci':
        model = dlinear(kwargs['num_nodes'], kwargs['input_len'], kwargs['output_len'])
    elif model_type == 'patchmlp':
        model = patchmlp(kwargs['num_nodes'], kwargs['input_len'], kwargs['output_len'])
    elif model_type == 'hkst':
        model = hkst(kwargs['num_nodes'], kwargs['input_len'], kwargs['output_len'], kwargs['gso'])

    elif model_type == 'patchtst' or model_type == 'patchtst_ci':
        model = patchtst(kwargs['num_nodes'], kwargs['input_len'], kwargs['output_len'])
    elif model_type == 'itransformer':
        model = itransformer(kwargs['num_nodes'], kwargs['input_len'], kwargs['output_len'])

    elif model_type == 'gwnet':
        from .gwnet.model import GWNet
        model = GWNet(kwargs['num_nodes'], kwargs['input_len'], kwargs['output_len'],
                      kwargs['gso'], kwargs['device'])
    elif model_type == 'stwave':
        model = stwave(kwargs['num_nodes'], kwargs['input_len'], kwargs['output_len'],
                       kwargs['train_loader'], kwargs['gso_path'])
    elif model_type == 'stdn':
        model = stdn(kwargs['num_nodes'], kwargs['input_len'], kwargs['output_len'], kwargs['gso_path'])
    elif model_type == 'staeformer':
        model = staeformer(kwargs['num_nodes'], kwargs['input_len'], kwargs['output_len'])
    elif model_type == 'pdformer':
        model = pdformer(kwargs['num_nodes'], kwargs['input_len'], kwargs['output_len'])
    # elif model_type == 'dstmamba':
    #     model = dstmamba(kwargs['num_nodes'], kwargs['input_len'], kwargs['output_len'])

    else:
        raise ValueError(f"Unknown model type {model_type}")
    return model
