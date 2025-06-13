#!/usr/bin/env python3
"""
Simple 3D Word Embeddings Visualization
Minimal version using only matplotlib for 3D visualization
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def load_data():
    """Load embeddings and vocabulary"""
    print("Loading embeddings and vocabulary...")
    
    # Load embeddings
    embeddings = np.load('data/embeddings.npy')
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Load word to index mapping
    with open('data/word_to_idx.pkl', 'rb') as f:
        word_to_index = pickle.load(f)
    print(f"Vocabulary size: {len(word_to_index)}")
    
    return embeddings, word_to_index

def main():
    # Load data
    embeddings, word_to_index = load_data()
    
    # Words to visualize
    viz_words = [
        'dog', 'cat', 'bird', 'fish', 'horse', 'cow', 'pig',
        'red', 'blue', 'green', 'yellow', 'black', 'white',
        'one', 'two', 'three', 'four', 'five', 'six',
        'america', 'england', 'france', 'china', 'city', 'country',
        'computer', 'internet', 'digital', 'software', 'technology',
        'happy', 'sad', 'love', 'angry', 'fear', 'joy'
    ]

    # Word categories
    word_categories = {
        'Animals': ['dog', 'cat', 'bird', 'fish', 'horse', 'cow', 'pig'],
        'Colors': ['red', 'blue', 'green', 'yellow', 'black', 'white'],
        'Numbers': ['one', 'two', 'three', 'four', 'five', 'six'],
        'Places': ['america', 'england', 'france', 'china', 'city', 'country'],
        'Tech': ['computer', 'internet', 'digital', 'software', 'technology'],
        'Emotions': ['happy', 'sad', 'love', 'angry', 'fear', 'joy']
    }

    # Filter available words
    available_words = [word for word in viz_words if word in word_to_index]
    print(f"Found {len(available_words)} words in vocabulary")
    
    if len(available_words) < 10:
        print("Not enough words for visualization")
        return
    
    # Get embeddings for available words
    viz_embeddings = np.array([embeddings[word_to_index[word]] for word in available_words])
    print(f"Visualization embeddings shape: {viz_embeddings.shape}")

    # 3D t-SNE
    print("Computing 3D t-SNE...")
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(available_words)-1))
    embeddings_3d = tsne.fit_transform(viz_embeddings)

    # Create 3D plot
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Color scheme
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot points
    for i, word in enumerate(available_words):
        x, y, z = embeddings_3d[i]
        
        # Find category color
        color = 'black'
        for cat_name, cat_words in word_categories.items():
            if word in cat_words:
                cat_index = list(word_categories.keys()).index(cat_name)
                color = colors[cat_index % len(colors)]
                break
        
        ax.scatter(x, y, z, c=color, s=100, alpha=0.7)
        ax.text(x, y, z, word, fontsize=9)

    ax.set_title('3D Word Embeddings (t-SNE)', fontsize=16)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')

    # Legend
    legend_elements = []
    for i, (cat_name, _) in enumerate(word_categories.items()):
        color = colors[i % len(colors)]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=8, label=cat_name))
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    
    # Save t-SNE plot
    tsne_filename = 'embeddings_3d_tsne.png'
    plt.savefig(tsne_filename, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE visualization to: {tsne_filename}")
    plt.close()

    # 3D PCA
    print("Computing 3D PCA...")
    pca = PCA(n_components=3)
    embeddings_pca = pca.fit_transform(viz_embeddings)

    # Create 3D PCA plot
    fig2 = plt.figure(figsize=(15, 12))
    ax2 = fig2.add_subplot(111, projection='3d')

    # Plot PCA points
    for i, word in enumerate(available_words):
        x, y, z = embeddings_pca[i]
        
        # Find category color
        color = 'black'
        for cat_name, cat_words in word_categories.items():
            if word in cat_words:
                cat_index = list(word_categories.keys()).index(cat_name)
                color = colors[cat_index % len(colors)]
                break
        
        ax2.scatter(x, y, z, c=color, s=100, alpha=0.7)
        ax2.text(x, y, z, word, fontsize=9)

    ax2.set_title('3D Word Embeddings (PCA)', fontsize=16)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax2.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax2.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    
    # Save PCA plot
    pca_filename = 'embeddings_3d_pca.png'
    plt.savefig(pca_filename, dpi=300, bbox_inches='tight')
    print(f"Saved PCA visualization to: {pca_filename}")
    plt.close()

    print(f"PCA explained variance: {sum(pca.explained_variance_ratio_):.1%}")
    print(f"\n✅ All visualizations saved! Check the files:")
    print(f"   - {tsne_filename}")
    print(f"   - {pca_filename}")

if __name__ == "__main__":
    main() 