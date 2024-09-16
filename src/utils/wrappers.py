import torch
from torch.nn import Module
from torch.nn import functional as F
from accelerate import Accelerator

import os

from subtrees.AudioMAE.models_mae import MaskedAutoencoderViT
from features.imu_preprocessing import SpectrogramsGenerator

class WrapperModel(Module):

    def __init__(self,
        vit_backbone: MaskedAutoencoderViT,
        model_on_top,
        cfg,
        args,
        accelerator: Accelerator
        ):
        super(WrapperModel, self).__init__()
 
        self.vit_backbone = vit_backbone
        self.model_on_top = model_on_top
        self.mask_t_prob = args.mask_t_prob
        self.mask_f_prob = args.mask_f_prob
        self.accelerator = accelerator
        self.batch_size = args.batch_size
        
        self.num_sensors = 4
        self.num_chans = 3
        self.seconds = args.seconds
        self.Fs = cfg['spectrogram_params'][f'sec_{args.seconds}'][args.matrix_type]['sampling_rate']
        self.seconds_bin = args.seconds * self.Fs * self.num_chans * self.num_sensors

        self.specgram_cfg = {
            'mean': torch.load(os.path.join(cfg['mean_std_path'], 'accl_mean.pt')),
            'std': torch.load(os.path.join(cfg['mean_std_path'], 'accl_std.pt')),
            **cfg['spectrogram_params'][f'sec_{args.seconds}'][args.matrix_type]
        }

        self.transforms_spectrogram = SpectrogramsGenerator(
            device=accelerator.device,
            **self.specgram_cfg
        )

        # self.transforms_spectrogram = SpectrogramsWEAR(
        #     training=False,
        #     max_seq_len=self.model_on_top.max_seq_len,
        #     cfg=cfg,
        #     args=args,
        #     device=accelerator.device,
        #     mask_t_prob=args.mask_t_prob,
        #     mask_f_prob=args.mask_f_prob,
        #     batch_size=args.batch_size
        # )


    # TODO: chech if wrapping is done correctly (WEAR).
    def forward(self, video_list):
        # (batched_inputs, batched_masks) = spectrogram_fn(video_list)
        (batched_inputs, batched_masks) = self.model_on_top.preprocessing(video_list)
        batched_inputs = self.forward_imu_vit(batched_inputs)

        #TODO: reshape x to fit the model_on_top
        x = (video_list, batched_inputs, batched_masks)
        x = self.model_on_top(x, need_preprocess=False)
        return x

    def forward_imu_vit(self, batched_inputs):
        '''
        forward the input through the Vision Transformer model and reshape to adapt to the model_on_top
        
        '''
        B, C, T = batched_inputs.shape
        tmp_tot = batched_inputs[:, :self.seconds_bin, :].detach()
        tmp_tot = tmp_tot.permute(0, 2, 1) # from (B, C, T) to (B, T, C)
        
        map_fn = torch.vmap(self.transforms_spectrogram, in_dims=0, randomness='same')
        imu_extracted_tot = None

        for vid, tmp in enumerate(tmp_tot):
            tmp = tmp.reshape(T // self.batch_size, self.batch_size, self.num_sensors * self.num_chans, self.seconds * self.Fs)
            imu_extracted = None
            for i, batch in enumerate(tmp):
                    with self.accelerator.autocast():
                        x = map_fn(batch)
                        b, c, h, w = x.shape
                        x = x.reshape(b, self.num_chans, self.num_sensors, h,  w)
                        x = self.vit_backbone.forward_embeds(x, mask_t_prob=self.mask_t_prob, mask_f_prob=self.mask_f_prob)
                        x = self.vit_backbone.forward_transformer(x)
                        if imu_extracted is None:
                            imu_extracted = x
                        else:
                            imu_extracted = torch.cat((imu_extracted, x), dim=0)
                        # del x
                        # torch.cuda.empty_cache()
            imu_extracted_tot = torch.cat((imu_extracted_tot.unsqueeze(0), imu_extracted.unsqueeze(0)), dim=0) if imu_extracted_tot is not None else imu_extracted

        return imu_extracted_tot.permute(0, 2, 1)
