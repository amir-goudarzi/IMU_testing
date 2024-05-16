import numpy as np
import pandas as pd
import os

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt



def load_embeddings(src_dir: str, filename: str, top_labels=0) -> tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings from a file and convert it into numpy.
    """
    embeddings = pd.read_pickle(os.path.join(src_dir, filename))

    if top_labels > 0:
        top_cls = embeddings['target'].value_counts().head(top_labels).index.values
        embeddings = embeddings[embeddings['target'].isin(top_cls)]
    embeddings_val = np.vstack(embeddings['feature_embeddings'].to_numpy())
    labels = np.vstack(embeddings['target'].to_numpy()).flatten()

    return embeddings_val, labels


def plot_TSNE(
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_components: int = 2,
        perplexity=30,
        n_iter=300,
        random_state=42,
        s=None,
        cmap='viridis',
        save_dir = None,
        save=False,
        filename='tsne_embeddings') -> None:
    """
    Plot TSNE embeddings.
    """

    if save_dir is None:
        save_dir = os.path.join('images', 'feature_analysis', 'tsne_embeddings')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        assert os.path.exists(save_dir), f"Directory {save_dir} does not exist."

    tsne_embeddings = gen_TSNE(embeddings, labels, n_components, perplexity, n_iter, random_state)
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idxs = np.where(labels == label)[0]
        tsne_embeddings_label = tsne_embeddings[idxs, :]
        if s is not None:
            plt.scatter(tsne_embeddings_label[:, 0], tsne_embeddings_label[:, 1], s=s, cmap=cmap, label=f'Class {label}')
        else:
            plt.scatter(tsne_embeddings_label[:, 0], tsne_embeddings_label[:, 1], cmap=cmap, label=f'Class {label}')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
    if save:
        plt.savefig(os.path.join(save_dir, f'{filename}.svg'), bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f'{filename}.png'), bbox_inches='tight')
    else:
    
        plt.show()


def plot_PCA(
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_components: int = 2,
        random_state=42,
        s=None,
        cmap='viridis',
        save_dir = None,
        save=False,
        filename='pca_embeddings') -> None:
    """
    Plot PCA embeddings.
    """

    if save_dir is None:
        save_dir = os.path.join('images', 'feature_analysis', 'pca_embeddings')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        assert os.path.exists(save_dir), f"Directory {save_dir} does not exist."

    pca_embeddings = gen_PCA(embeddings, labels, n_components, random_state)
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idxs = np.where(labels == label)[0]
        pca_embeddings_label = pca_embeddings[idxs, :]
        if s is not None:
            plt.scatter(pca_embeddings_label[:, 0], pca_embeddings_label[:, 1], s=s, cmap=cmap, label=f'Class {label}')
        else:
            plt.scatter(pca_embeddings_label[:, 0], pca_embeddings_label[:, 1], cmap=cmap, label=f'Class {label}')
    plt.legend(loc='center left')

    if save:
        plt.savefig(os.path.join(save_dir, f'{filename}.svg'))
        plt.savefig(os.path.join(save_dir, f'{filename}.png'))
    else:
        plt.show()


def gen_TSNE(embeddings: np.ndarray, labels: np.ndarray, n_components: int = 2, perplexity=30, n_iter=300, random_state=42) -> np.ndarray:
    """
    Generate TSNE embeddings from the input embeddings.
    """
    tsne = TSNE(n_components=n_components, perplexity=30, n_iter=300, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)

    return tsne_embeddings


def gen_PCA(embeddings: np.ndarray, labels: np.ndarray, n_components: int = 2, random_state=42) -> np.ndarray:
    """
    Generate PCA embeddings from the input embeddings.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_embeddings = pca.fit_transform(embeddings)

    return pca_embeddings
