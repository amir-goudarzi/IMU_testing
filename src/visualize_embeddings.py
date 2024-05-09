from visualization.feature_analysis import plot_TSNE, load_embeddings
import os

if __name__ == '__main__':
    embeddings_loc = os.path.join('data', 'EPIC-KITCHENS', 'IMU', 'audio_mae', 'cls_token')
    filename = 'feature_embeddings.pkl'
    embeddings, labels = load_embeddings(embeddings_loc, filename, top_labels=10)

    plot_TSNE(embeddings, labels, n_components=2, perplexity=30, n_iter=300, random_state=42)