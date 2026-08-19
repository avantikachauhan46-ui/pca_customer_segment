import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_kmeans(X_pca, df, n_clusters=3,
               output_path="outputs/cluster_visualization.png"):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)

    result = df.copy()
    result["Cluster"] = labels
    result.to_csv("outputs/customer_clusters.csv", index=False)

    plt.figure(figsize=(9, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.scatter(model.cluster_centers_[:, 0],
                model.cluster_centers_[:, 1],
                marker="X", s=200)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Customer Segmentation | Silhouette Score: {score:.3f}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return model, labels, score
