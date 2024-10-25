import seaborn as sns
import pandas as pd
import argparse
import os

import matplotlib.pyplot as plt

from copy import deepcopy


PATH_DF = {
    'actionformer': '../logs/actionformer/actionformer_general_mean.csv',
    'tridet': '../logs/tridet/tridet_general_mean.csv',
}
PATH_DF = {
    'actionformer': 'src/subtrees/wear/logs/actionformer/actionformer_general_mean.csv',
    'tridet': 'src/subtrees/wear/logs/tridet/tridet_general_mean.csv',
}

MAP_METRIC = {
    'P': 'Precision',
    'R': 'Recall',
    'F1': 'F1 Score',
    'avg_mAP': 'Average mAP'
}

MODELS_NAMES = {
    'inertial': 'inertial (RAW IMU features)',
    'combined': 'combined (RAW IMU features)',
    'inertial_mae_10epochs': 'inertial (MAE IMU features, IMU Ego-Exo4D->WEAR)',
    'combined_mae_10epochs': 'combined (MAE IMU features, IMU Ego-Exo4D->WEAR)',
    'inertial_mae_i3d_pt': 'inertial (MAE features, IMU+I3D Ego-Exo4D->WEAR)',
    'combined_mae_i3d_pt': 'combined (MAE features, IMU+I3D Ego-Exo4D->WEAR)',
    'inertial_mae_scratch_fixed': 'inertial (MAE features, IMU WEAR->WEAR)',
    'combined_mae_scratch_fixed': 'combined (MAE features, IMU WEAR->WEAR)',
}

def map_models_names(df, model_name='actionformer'):
    df['model'] = df['model'].map(lambda x: MODELS_NAMES[x.replace(model_name, '')])
    return df

def lineplot_score_thresh(args, model='actionformer'):
    df = pd.read_csv(PATH_DF[model])
    df_model_name = df['model'].map(lambda x: x.replace('single_gpu', model))
    df['model'] = df_model_name
    df = df.drop(['Unnamed: 0'], axis=1)
    df = df.dropna()
    # df = df.sort_values(by='threshold')
    df = df.reset_index(drop=True)

    labels = []
    selection = []

    num_distinct_thresholds = len(df['threshold'].unique())
    for exp in args.experiment_type[0].split(' '):
        labels.append(f'{MODELS_NAMES[exp]}')
        selection.append(f'{model}{exp}')

    df = df[df['model'].isin(selection)]
    df = map_models_names(df, model_name=model)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    metric = MAP_METRIC[args.metric]

    
    
    sns.lineplot(
        data=df,
        x='threshold', y=args.metric,
        hue='model',
        style='model',
        markers=True,
        dashes=False,
    )

    plt.legend(title=f'Configurations', loc="upper center", bbox_to_anchor=(0.5, -0.2))
    plt.title(f'{model} {metric} for each score threshold')
    plt.xlabel('Threshold')
    plt.ylabel(f'{metric}')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, model, f'{model}_{args.experiment_name}_{args.metric}_score_thresh.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot a selected metric depending on the threshold')
    parser.add_argument('--model', type=str, default='actionformer', help='choose general model')
    parser.add_argument('--metric', type=str, default='avg_mAP', help='Metric to plot', choices=['P', 'R', 'F1', 'avg_mAP'])
    parser.add_argument('--experiment_type', default='combined_mae_10epochs', nargs='*')
    parser.add_argument('--experiment_name', type=str, help='Threshold to plot')
    parser.add_argument('--save_path', type=str, default='src/subtrees/wear/logs', help='Path to save the plot')
    args = parser.parse_args()
    lineplot_score_thresh(args, args.model)

    choices=[
        'inertial',
        'combined',
        'inertial_mae_10epochs',
        'combined_mae_10epochs',
        'inertial_mae_i3d_pt',
        'combined_mae_i3d_pt',
        'inertial_mae_scratch_fixed',
        'combined_mae_scratch_fixed',
    ]