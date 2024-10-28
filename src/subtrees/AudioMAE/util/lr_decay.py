# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers
    

def linprob_parse(model, no_weight_decay_list):
    """
    Parse the model to get the linear probe layer
    """
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True
        
    return no_weight_decay_list

def linprob_parse_omni(model, no_weight_decay_list):
    """
    Parse the model to get the linear probe layer
    """
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True
    for _, p in model.omni_classifier.named_parameters():
        p.requires_grad = True
        
    return no_weight_decay_list

def linprob_parse_omni_late_fusion(model, no_weight_decay_list):
    """
    Parse the model to get the linear probe layer
    """
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True
    for _, p in model.omni_classifier.named_parameters():
        p.requires_grad = True
    for _, p in model.late_fusion.named_parameters():
        p.requires_grad = True
        
    return no_weight_decay_list

def linprob_parse_interfusion(model, no_weight_decay_list):
    """
    Parse the model to get the linear probe layer
    """
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True
    for _, p in model.interfuse.named_parameters():
        p.requires_grad = True
        
    return no_weight_decay_list

def wear_freeze_decoder(model, no_weight_decay_list):
    """
    Freeze the decoder part of the model
    """
    for _, p in model.named_parameters():
        p.requires_grad = True
    # for _, p in model.decoder_pos_embed.named_parameters():
    #     p.requires_grad = False
    for _, p in model.decoder_blocks.named_parameters():
        p.requires_grad = False
    for _, p in model.decoder_norm.named_parameters():
        p.requires_grad = False
    for _, p in model.decoder_pred.named_parameters():
        p.requires_grad = False
        
    return no_weight_decay_list