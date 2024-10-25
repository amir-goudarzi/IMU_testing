import torch.nn as nn
import torch
from safetensors.torch import load_file
from timm.models.layers import trunc_normal_

import os
import numpy as np

from subtrees.AudioMAE.util.patch_embed import PatchEmbed3D_new
from audiomae_ft import modeling, load_model
import subtrees.AudioMAE.models_mae as models_mae

def load_vit3d_model(seconds,
        matrix_type,
        cfg):
    '''
    function to load the 3D Vision Transformer model
    '''

    specgram_cfg = cfg['spectrogram_params'][f'sec_{seconds}'][matrix_type]
    model_dict = {attr: getattr(models_mae, attr) for attr in dir(models_mae)}
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
    model = model_dict[model_name]( img_size=video_size, **cfg['model'] )

    return model

def load_mae_model(finetune, eval, model):
    # accelerator.load_state(finetune)
    checkpoint_model = load_file(os.path.join(finetune, 'model.safetensors'))
    print("Load pre-trained checkpoint from: %s" % finetune)
    state_dict = model.state_dict()

    patchembed3d = checkpoint_model['patch_embed.proj.weight']
    checkpoint_model['patch_embed.proj.weight'] = patchembed3d.unsqueeze(2)
    # checkpoint_model['patch_embed.proj.weight'] = patchembed3d.unsqueeze(2).repeat(1,1,4,1,1)
  
    checkpoint_model['pos_embed'] = tile_positional_embeddings(checkpoint_model['pos_embed'][0][1:], model.patch_embed.num_patches)
    checkpoint_model['pos_embed'] = torch.cat((checkpoint_model['cls_token'], checkpoint_model['pos_embed']), dim=1)

    decoder_cls_token = checkpoint_model['decoder_pos_embed'][:, :1, :]
    checkpoint_model['decoder_pos_embed'] = tile_positional_embeddings(checkpoint_model['decoder_pos_embed'][0][1:], model.decoder_pos_embed.shape[1]-1)
    checkpoint_model['decoder_pos_embed'] = torch.cat((decoder_cls_token, checkpoint_model['decoder_pos_embed']), dim=1)

    del checkpoint_model['decoder_embed.weight'], checkpoint_model['decoder_embed.bias']

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    model.decoder_embed.apply(model._init_weights)
    print(msg)

    # # manually initialize fc layer
    # if not eval:
    #     trunc_normal_(model.head.weight, std=2e-5)
    
    return model

def load_mae_model_2d(finetune, eval, model):
    # accelerator.load_state(finetune)
    if finetune:
      checkpoint_model = load_file(os.path.join(finetune, 'model.safetensors'))
      print("Load pre-trained checkpoint from: %s" % finetune)
      state_dict = model.state_dict()

      patchembed3d = checkpoint_model['patch_embed.proj.weight']
      checkpoint_model['patch_embed.proj.weight'] = checkpoint_model['patch_embed.proj.weight'].repeat(1, 4, 1, 1) / 4
      # checkpoint_model['patch_embed.proj.weight'] = patchembed3d.unsqueeze(2).repeat(1,1,4,1,1)

      # load pre-trained model
      del checkpoint_model['decoder_embed.weight'], checkpoint_model['decoder_embed.bias']

      checkpoint_model['decoder_pred.weight'] = checkpoint_model['decoder_pred.weight'].repeat(4, 1) / 4
      checkpoint_model['decoder_pred.bias'] = checkpoint_model['decoder_pred.bias'].repeat(4) / 4
      # load pre-trained model
      msg = model.load_state_dict(checkpoint_model, strict=False)
      model.decoder_embed.apply(model._init_weights)
    # # manually initialize fc layer
    # if not eval:
    #     trunc_normal_(model.head.weight, std=2e-5)
    
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

  return torch.tensor(restored_posemb_grid)