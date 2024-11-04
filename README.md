# HIMU-MAE: Exploiting Head-mounted Inertial Measurement Unit with Masked Autoencoders for Egocentric Vision

## 1. Environment setup

To create the environment, run this:

```
conda env create -f env.yaml 
conda activate himu-mae
```



## 2. Datasets

You can find the datasets code in `src/data` directory. Each dataset have been registered via decorators, so you can have easier life. To use them, you can leverage the `make_dataset()` function in `src/data/dataset.py`, by passing it the Key ID (e.g. name) of the dataset and all its necessary arguments.

Example:

```
from data.dataset import make_dataset

train_set = make_dataset(
                name='egoexo4d', # Key ID
                is_pretrain=True,
                **cfg_args_train # Includes both pre-processing and dataset args
            )
```

The already registered datasets are:

| Dataset | Key ID |
| ------- | ---- |
| Ego-Exo4D | egoexo4d |
| WEAR (for SSL) | wear_ssl |


You can include a new dataset by adding the following line as shown below:
```
from data.dataset import register_dataset

@register_dataset("egoexo4d")
class EgoExo4D(Dataset):
    def __init__(...
```
### 2.1 Pre-processing

For pre-processing we applied the same strategy as in [EVI-MAE](https://arxiv.org/html/2407.06628v1). We have directly contacted the authors for spectrograms generator.

### 2.2 Ego-Exo4D

First, if the data are not already available in the machine, you have to follow the steps illustrated [here](https://docs.ego-exo4d-data.org/getting-started/#download-dataset) to download each part.

Pre-extracted video features with Omnivore are available according to the [official website](https://docs.ego-exo4d-data.org/data/features/). The IMU readings have been extracted from VRS files according to [tutorial](https://github.com/facebookresearch/Ego4d/blob/main/notebooks/egoexo/EgoExo_Aria_Data_Tutorial.ipynb), but I've modified it in `notebooks/egoexo4d.ipynb`. 

Code available in `src/data/egoexo4d.py`.

The implementation uses only left-IMU, but you can use both of them by adjusting the code. Please note that left-IMU has been sampled at 800Hz, while the right one at 1kHz.


### 2.3 WEAR SSL

Code available in `src/data/wear_dataset_ssl.py`.

To download it please follow the instructions [here](https://github.com/mariusbock/wear).

The annotations are the same as in `WEAR/annotations`: they are divided in three splits. 


## 3. Self-Supervised Pre-Training with Masked Autoencoders (MAE)

We have exploited [Audio-MAE](https://github.com/facebookresearch/AudioMAE) code, but it leverages PyTorch Distributed Data Parallel, and it has been set up to run in that way. We leveraged [HuggingFace's Accelerate](https://huggingface.co/docs/accelerate/index) on top of that code, but it could be simplified by bringing useful stuff from `src/audiomae_pt.py` to `src/dist_pretrain_accelerate.py`, as well as `src/audiomae_ft.py` to `src/dist_ft_accelerate.py`.

To run the pre-training, it is necessary to create a config (yaml) and a bash script. You can consider `configs/IMU-MAE/egoexo4d_accl.yaml` and `scripts/AudioMAE/accelerator_pretrain.sh` as examples.

IMU-only:
- cfg: `configs/IMU-MAE/egoexo4d_accl.yaml`
- script: `scripts/AudioMAE/accelerator_pretrain.sh`

IMU + Omnivore:
- cfg: `configs/IMU-MAE/egoexo4d_accl_omni_pt.yaml`
- script: `scripts/AudioMAE/accelerator_pretrain_omni.sh`

IMU visualization (no norm pix loss):
- cfg: `configs/IMU-MAE/egoexo4d_accl_visualization.yaml`
- script: `scripts/AudioMAE/accelerator_pretrain_visualization.sh`

## 4. Linear Probing (LP) & Fine-Tuning (FT)

It consists of two different experiments:
1. **Late Fusion**: each modality stream outputs logits, then they've been summed;
2. **Intermediate Fusion** (the best): each modality stream outputs feature embeddings, then they've been concatenated and passed as input to a projector for Linear Probing, or into a MLP for Fine-Tuning.

- IMU + Omni LP - Late Fusion: `scripts/AudioMAE/accelerator_omni_linprob.sh`
- IMU + Omni LP - Intermediate Fusion: `scripts/AudioMAE/accelerator_omni_interfuse.sh`
- IMU + Omni FT - Intermediate Fusion: `scripts/AudioMAE/accelerator_omni_ft_late_interfusion.sh`

## 5. WEAR experiments

### 5.1 Continued Pre-Training

Here the following pretrain:
- IMU (with Ego-Exo4D pretrain): `scripts/wear/accelerator_pretrain_transfer_wear.sh`
- IMU (from scratch): `scripts/wear/accelerator_pretrain_wear.sh`
- IMU + I3D (with Ego-Exo4D pretrain): `scripts/wear/accelerator_pretrain_transfer_wear_combined.sh`

### 5.2 Extract features

The reference is `scripts/wear/extract_feats.sh` (it's not working at the moment, path issues), but I've used "extract feats MAE - WEAR" from `.vscode/launch.json` by changing the values. Better to fix the scripts if you can.

They will be saved in a sub-path of `WEAR/processed/combined_features/120_frames_60_stride/mae/` (mind the dataset path!), but check the code if you want to insert more features.

### 5.3 Temporal Action Localisation

The baseline is from WEAR github code. To run the experiments you need two elements:
- config: they are in `src/subtrees/wear/configs/120_frames_60_stride`; if you want to use a new feature type, you need to create one of them from the baseline `src/subtrees/wear/configs/120_frames_60_stride/actionformer_inertial.yaml` and `src/subtrees/wear/configs/120_frames_60_stride/actionformer_combined.yaml` (and the TriDet equivalent) by changing the features path.
- bash

Here are the following scripts. I'm reporting only actionformer inertial, but take it as baseline for Actionformer combined and TriDet (inertial and combined).

Actionformer Inertial:
- IMU (with Ego-Exo4D pretrain): `scripts/wear/actionformer_inertial_mae.sh`
- IMU (from scratch): `scripts/wear/actionformer_inertial_mae_fromscratch.sh`
- IMU + I3D (with Ego-Exo4D pretrain): `scripts/wear/actionformer_inertial_mae_i3d.sh`

