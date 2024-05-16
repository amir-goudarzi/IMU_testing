from visualization.feature_analysis import plot_TSNE, load_embeddings
import os

if __name__ == '__main__':
    embedding_type = 'global_pool'
    mask=0.5
    embeddings_loc = os.path.join('data', 'EPIC-KITCHENS', 'IMU', 'audio_mae', embedding_type)
    filename = f'feature_embeddings_mask({mask}).pkl'
    embeddings, labels = load_embeddings(embeddings_loc, filename, top_labels=10)

    plot_TSNE(embeddings, labels, n_components=2, perplexity=30, n_iter=300, random_state=42)