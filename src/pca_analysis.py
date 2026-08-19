import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def run_pca(X_scaled, output_path="outputs/pca_visualization.png"):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total explained variance:", pca.explained_variance_ratio_.sum())

    plt.figure(figsize=(9, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Visualization")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return pca, X_pca
