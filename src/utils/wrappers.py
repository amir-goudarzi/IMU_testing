import sys, os

sys.path.append(os.path.join('src'))
# from audiomae_pretrain import modeling
from audiomae_ft import modeling
from utils.os_utils import load_config
from features.imu_preprocessing import SpectrogramsGenerator

import torch
from torch.nn import Module

class WrapperMAE(Module):

    def __init__(self,
        config_mae: str,
        seconds: int,
        matrix_type: str,
        checkpoint: str,
        eval: True):
        super(WrapperMAE, self).__init__()
        # Load config
        self.config = load_config(config_mae)
        spectrogram_cfg = self.config['spectrogram_params'][f'sec_{seconds}'][matrix_type]
        # Load model
        # self.mae_backbone = modeling(
        #     seconds=seconds,
        #     matrix_type=matrix_type,
        #     cfg=self.config,
        #     audio_exp=self.config['model']['audio_exp']
        # )
        self.mae_backbone = modeling(
            seconds=seconds,
            matrix_type=matrix_type,
            cfg=self.config,
            audio_exp=self.config['model']['audio_exp'],
            eval=eval,
            finetune=checkpoint
        )
        load_checkpoint = torch.load(checkpoint)['model']
        if self.config['model']['pretrain']:
            self.mae_backbone.load_state_dict(load_checkpoint)

        if eval:
            self.mae_backbone.eval()
            for param in self.mae_backbone.parameters():
                param.requires_grad = False

        self.transforms = SpectrogramsGenerator(
            window_size=spectrogram_cfg['window_size'],
            overlap_in_s=spectrogram_cfg['overlap_in_s'],
            n_fft=spectrogram_cfg['n_fft'],
            hop_length=spectrogram_cfg['hop_length'],
            sampling_rate=self.config['dataset']['sampling_rate'],
            downsampling_rate=spectrogram_cfg['downsampling_rate'],
            resizes=spectrogram_cfg['resizes']
        )
        

    def forward(self, x):
        x = self.transforms(x)
        # x, _, _, _ = self.mae_backbone.forward_encoder(x, self.config['model']['mask_ratio'], self.config['model']['mask_2d'])
        x = self.mae_backbone(x)
        return x
