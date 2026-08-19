import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="wide")
st.title("📊 Customer Segmentation Dashboard")
st.caption("PCA + K-Means clustering")

uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
df = pd.read_csv(uploaded) if uploaded else pd.read_csv("data/customers.csv")

features = [
    "Age", "AnnualIncome", "SpendingScore",
    "PurchaseFrequency", "WebsiteVisits"
]

st.subheader("Dataset Preview")
st.dataframe(df.head(10), width="stretch")

if df[features].isnull().sum().sum():
    st.error("Missing values detected. Clean the dataset before clustering.")
    st.stop()

X_scaled = StandardScaler().fit_transform(df[features])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

st.subheader("PCA Analysis")
a, b, c = st.columns(3)
a.metric("Original Features", len(features))
b.metric("PCA Components", 2)
c.metric("Variance Retained", f"{pca.explained_variance_ratio_.sum()*100:.1f}%")

k = st.slider("Choose number of clusters (K)", 2, 8, 3)
model = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = model.fit_predict(X_pca)
score = silhouette_score(X_pca, labels)

st.metric("Silhouette Score", f"{score:.3f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
ax.scatter(model.cluster_centers_[:, 0],
           model.cluster_centers_[:, 1],
           marker="X", s=200)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("Customer Segmentation using PCA + K-Means")
st.pyplot(fig)

result = df.copy()
result["Cluster"] = labels

st.subheader("Cluster Summary")
st.dataframe(result.groupby("Cluster")[features].mean().round(2),
             width="stretch")

st.download_button(
    "💾 Download Clustered Data",
    result.to_csv(index=False).encode("utf-8"),
    "customer_clusters.csv",
    "text/csv"
)
