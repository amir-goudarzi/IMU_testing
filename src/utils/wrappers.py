import torch
from torch.nn import Module
from torch.nn import functional as F
from accelerate import Accelerator

import os

from subtrees.AudioMAE.models_vit import VisionTransformer
from features.imu_preprocessing import SpectrogramsGenerator

class WrapperModel(Module):

    def __init__(self,
        vit_backbone: VisionTransformer,
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

        specgram_cfg = {
            'mean': torch.load(os.path.join(cfg['mean_std_path'], 'accl_mean.pt')),
            'std': torch.load(os.path.join(cfg['mean_std_path'], 'accl_std.pt')),
            **cfg['spectrogram_params'][f'sec_{args.seconds}'][args.matrix_type]
        }

        self.transforms_spectrogram = SpectrogramsGenerator(
            device=accelerator.device,
            **specgram_cfg
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
        x = (batched_inputs, batched_masks)
        x = self.model_on_top(x, need_preprocess=False)
        return x

    def forward_imu_vit(self, batched_inputs):
        '''
        forward the input through the Vision Transformer model and reshape to adapt to the model_on_top
        
        '''
        B, C, T = batched_inputs.shape
        tmp_tot = batched_inputs[:, :self.seconds_bin, :].detach()
        tmp_tot = tmp_tot.permute(0, 2, 1) # from (B, C, T) to (B, T, C)
        
        imu_extracted = imu_extracted = torch.empty(B, T // self.batch_size, self.batch_size, self.vit_backbone.embed_dim, device='cpu')
        map_fn = torch.vmap(self.transforms_spectrogram, in_dims=0, randomness='same')

        for vid, tmp in enumerate(tmp_tot):
            tmp = tmp.reshape(T // self.batch_size, self.batch_size, self.num_sensors * self.num_chans, self.seconds * self.Fs)

            for i, batch in enumerate(tmp):
                with self.accelerator.autocast():
                    x = map_fn(batch)
                    b, c, h, w = x.shape
                    x = x.reshape(b, self.num_chans, self.num_sensors, h,  w)
                    x = self.vit_backbone(x, mask_t_prob=self.mask_t_prob, mask_f_prob=self.mask_f_prob)
                    imu_extracted[vid, i, :, :] = x.cpu()
                    del x
                    torch.cuda.empty_cache()
            imu_extracted = imu_extracted.reshape(B, T, imu_extracted.shape[-1]).permute(0, 2, 1)
        batched_inputs = torch.cat((batched_inputs[:, self.seconds_bin:, :], imu_extracted), dim=1)
        return x

    
class SpectrogramsWEAR(object):
    def __init__(self,
        training: bool,
        max_seq_len: int,
        cfg,
        args,
        max_div_factor=None,
        device=None,
        mask_t_prob=.0,
        mask_f_prob=.0,
        batch_size=1
        ):

        self.training = training
        self.max_seq_len = max_seq_len
             
        specgram_cfg = {
            'mean': torch.load(os.path.join(cfg['mean_std_path'], 'accl_mean.pt')),
            'std': torch.load(os.path.join(cfg['mean_std_path'], 'accl_std.pt')),
            **cfg['spectrogram_params'][f'sec_{args.seconds}'][args.matrix_type]
            }

        self.transforms_spectrogram = SpectrogramsGenerator(
            device=device,
            **specgram_cfg
        )
        self.mask_t_prob = mask_t_prob
        self.mask_f_prob = mask_f_prob
        self.batch_size = batch_size
        self.num_sensors = 4
        self.num_chans = 3
        self.seconds = args.seconds
        self.Fs = cfg['spectrogram_params'][f'sec_{args.seconds}'][args.matrix_type]['sampling_rate']

        self.seconds_bin = args.seconds * self.Fs * self.num_chans * self.num_sensors
        if max_div_factor is not None:
            self.max_div_factor = max_div_factor

    def __call__(self, batched_inputs: torch.Tensor):
        return self.imu_features(batched_inputs)
    
    def imu_features(self, batched_inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the audio features from the MAE model.
        It calculates spectrograms for each input, then forward the MAE model, which
        outputs the new feature embeddings instead of RAW ones.
        """
        B, C, T = batched_inputs.shape
        tmp = batched_inputs[:, :self.seconds_bin, :].detach()
        tmp = tmp.permute(0, 2, 1) # from (B, C, T) to (B, T, C)
        tmp = tmp.reshape(B * T, self.num_sensors * self.num_chans, self.seconds * self.Fs)

        map_fn = torch.vmap(self.transforms_spectrogram, in_dims=0)
        
        audio_feats = map_fn(tmp)
        return audio_feats

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs
        batched_masks = batched_masks.unsqueeze(1)

        return batched_inputs, batched_masks
    