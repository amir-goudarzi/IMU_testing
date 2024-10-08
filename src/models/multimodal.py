from typing import Dict, List
import torch.nn as nn
from torchmultimodal.models.late_fusion import LateFusion
from torchmultimodal.modules.fusions.concat_fusion import ConcatFusionModule
from torchmultimodal.modules.layers.mlp import MLP

from subtrees.AudioMAE.models_vit import VisionTransformer

def create_late_fusion(encoders: nn.ModuleDict, in_dim: int, num_classes: int, hidden_dims: List[int]) -> LateFusion:
    '''
    function to create the late fusion model
    '''
    # define the multimodal model
    fusion = ConcatFusionModule()
    classifier = MLP(
        in_dim=in_dim,
        out_dim=num_classes,
        hidden_dims=hidden_dims,
        activation=nn.GELU,
        normalization=nn.BatchNorm1d
    )
    model = LateFusion(
        encoders=encoders,
        fusion_module=fusion,
        head_module=classifier
    )

    return model

def custom_late_fusion(model: VisionTransformer, in_dim: int, num_classes: int, hidden_dims: List[int]) -> LateFusion:
    '''
    function to create the late fusion model
    '''
    # define the multimodal model
    fusion = ConcatFusionModule()
    model.late_fusion = MLP(
        in_dim=in_dim,
        out_dim=num_classes,
        hidden_dims=hidden_dims,
        activation=nn.GELU,
        normalization=nn.BatchNorm1d
    )
    del model.head, model.omni_classifier

    return model