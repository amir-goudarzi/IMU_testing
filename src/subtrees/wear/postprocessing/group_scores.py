import pandas as pd
import os

DIRS = {
    'actionformer': [
    # '../logs/actionformer/single_gpu/camera',
    '../logs/actionformer/single_gpu/inertial',
    '../logs/actionformer/single_gpu/combined',
    # '../logs/actionformer/single_gpu/inertial_mae',
    # '../logs/actionformer/single_gpu/combined_mae',
    # '../logs/actionformer/combined_mae_fixed',
    # '../logs/actionformer/inertial_mae_fixed',
    '../logs/actionformer/combined_mae_scratch_fixed',
    '../logs/actionformer/inertial_mae_scratch_fixed',
    '../logs/actionformer/combined_mae_10epochs',
    '../logs/actionformer/inertial_mae_10epochs',
    '../logs/actionformer/inertial_mae_i3d_pt',
    '../logs/actionformer/combined_mae_i3d_pt',
                 ],
    'tridet': [
    # '../logs/tridet/single_gpu/camera',
    '../logs/tridet/single_gpu/inertial',
    '../logs/tridet/single_gpu/combined',
    # '../logs/tridet/single_gpu/inertial_mae',
    # '../logs/tridet/single_gpu/combined_mae',
    '../logs/tridet/combined_mae_scratch_fixed',
    '../logs/tridet/inertial_mae_scratch_fixed',
    # '../logs/tridet/combined_mae_fixed',
    # '../logs/tridet/inertial_mae_fixed',
    '../logs/tridet/combined_mae_10epochs',
    '../logs/tridet/inertial_mae_10epochs',
    '../logs/tridet/inertial_mae_i3d_pt',
    '../logs/tridet/combined_mae_i3d_pt',
                 ],
}

def group_scores(model='actionformer', save_path = '../logs'):
    model_results_list = DIRS[model]

    general_df = None

    for path in model_results_list:
        tmp_df = pd.read_csv(f'{path}/{model}_mean.csv')
        if general_df is None:
            general_df = tmp_df
        else:
            general_df = pd.concat([general_df, tmp_df])
    general_df = general_df.round({'P': 2, 'R': 2, 'F1': 2, 'avg_mAP': 2})
    
    print(general_df)
    general_df.to_csv(os.path.join(save_path, model, f'{model}_general_mean.csv'), index=False)

if __name__ == '__main__':
    group_scores('actionformer')
    group_scores('tridet')
