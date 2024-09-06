import torch
from torch.nn import Module
from torch.nn import functional as F

from subtrees.AudioMAE.models_vit import VisionTransformer
from subtrees.wear.camera_baseline.actionformer.libs.modeling.meta_archs import PtTransformer
from features.imu_preprocessing import SpectrogramsGenerator

class WrapperModel(Module):

    def __init__(self,
        vit_backbone: VisionTransformer,
        model_on_top: PtTransformer,
        cfg
        ):
        super(WrapperModel, self).__init__()
 
        self.vit_backbone = vit_backbone
        self.model_on_top = model_on_top
        specgram_cfg = cfg['spectrogram']
        self.transforms_spectrogram = SpectrogramsGenerator(
            **specgram_cfg
        )

    # TODO: chech if wrapping is done correctly (WEAR).
    def forward(self, x):
        (batched_inputs, batched_masks) = self.preprocessing(x)
        batched_inputs = self.imu_features(batched_inputs)
        x = (batched_inputs, batched_masks)
        x = self.model_on_top(x)
        return x
    

    def imu_features(self, batched_inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the audio features from the MAE model.
        It calculates spectrograms for each input, then forward the MAE model, which
        outputs the new feature embeddings instead of RAW ones.
        """
        B, C, T = batched_inputs.shape
        tmp = batched_inputs[:, :self.model_on_top.seconds_bin, :].detach()
        tmp = tmp.permute(0, 2, 1) # from (B, C, T) to (B, T, C)

        tmp = tmp.reshape(-1, self.model_on_top.CH, self.model_on_top.T)
        tmp = self.transforms_spectrogram(tmp)

        x = self.vit_backbone(tmp)

        # Restore the original shape (i.e. include ViT embedding to the I3D features)
        x = torch.reshape(x, (B,T, x.shape[-1])).permute(0, 2, 1)
        audio_feats = torch.cat((x, batched_inputs[:, self.model_on_top.seconds_bin:, :]), dim=1)
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
            assert max_len <= self.model_on_top.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.model_on_top.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.model_on_top.max_seq_len:
                max_len = self.model_on_top.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.model_on_top.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.model_on_top.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.model_on_top.device)

        return batched_inputs, batched_masks
    
    def generate_spectrograms(self, x):
        """
        Generate spectrograms from the input audio.
        """
        