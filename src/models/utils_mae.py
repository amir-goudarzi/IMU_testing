import torch.nn as nn
import torch
from safetensors.torch import load_file
from timm.models.layers import trunc_normal_

import os
import numpy as np

from subtrees.AudioMAE.util.patch_embed import PatchEmbed3D_new
from audiomae_ft import modeling, load_model
import subtrees.AudioMAE.models_vit as models_vit

def load_vit3d_model(seconds,
        matrix_type,
        cfg,
        finetune,
        eval):
    '''
    function to load the 3D Vision Transformer model
    '''

    specgram_cfg = cfg['spectrogram_params'][f'sec_{seconds}'][matrix_type]
    model_dict = {attr: getattr(models_vit, attr) for attr in dir(models_vit)}
    model_name = cfg['model']['name'] + str(cfg['model'][matrix_type]['patch_size'][1])
    img_size = specgram_cfg['resizes']
    patch_size = cfg['model'][matrix_type]['patch_size']
    in_chans = cfg['model']['in_chans']
    emb_dim = cfg['model']['embed_dim']
    stride = (patch_size[0], patch_size[1], patch_size[2])
    video_size = (4, img_size[0], img_size[1])

    del cfg['model']['name']
    del cfg['model'][matrix_type]
    del cfg['model']['embed_dim']

    # model = model_dict[model_name]( **cfg['dataset_train'], **cfg['model'] )
    model = model_dict[model_name]( **cfg['model'] )


    model.patch_embed = PatchEmbed3D_new(video_size=video_size,
                                            patch_size=patch_size,
                                            in_chans=in_chans,
                                            embed_dim=emb_dim,
                                            stride=stride
                                            )
    num_patches = model.patch_embed.num_patches
    model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))

    return model

def load_mae_model(finetune, eval, model):
    # accelerator.load_state(finetune)
    checkpoint_model = load_file(os.path.join(finetune, 'model.safetensors'))
    print("Load pre-trained checkpoint from: %s" % finetune)
    state_dict = model.state_dict()

    patchembed3d = checkpoint_model['patch_embed.proj.weight']
    checkpoint_model['patch_embed.proj.weight'] = patchembed3d.unsqueeze(2).repeat(1,1,4,1,1)

    checkpoint_model['pos_embed'] = tile_positional_embeddings(checkpoint_model['pos_embed'][0], state_dict.patch_embed.num_patches + 1)


    if not eval:
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    # manually initialize fc layer
    if not eval:
        trunc_normal_(model.head.weight, std=2e-5)
    
    return model

def tile_positional_embeddings(restored_posemb_grid, n_tokens):
  """
  Implementation from https://github.com/google-research/scenic/blob/main/scenic/projects/vivit/model_utils.py#L249.
  Tile positional embeddings.

  Args:
    restored_posemb_grid: Positional embeddings from restored model. Shape is
      [n_restored_tokens, d]
    n_tokens: Number of tokens in the target model.

  Returns:
    positional embedding tiled to match n_tokens. Shape is [1, n_tokens, d]
  """

  num_repeats = int(n_tokens / len(restored_posemb_grid))
  restored_posemb_grid = np.concatenate(
      [restored_posemb_grid] * num_repeats, axis=0)
  restored_posemb_grid = np.expand_dims(restored_posemb_grid, axis=0)

  return restored_posemb_grid