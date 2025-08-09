import pandas as pd
import numpy as np
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    confusion_matrix,
    accuracy_score,
    adjusted_rand_score
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from pandas.plotting import parallel_coordinates

# Setup directories
fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)
summary_path = "segment_summary.csv"

# 1. Load and prepare data
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv"
df = pd.read_csv(url)

# Feature engineering
features = df.drop(columns=["Customer Id", "Address"])
features["Total Debt"] = features["Card Debt"] + features["Other Debt"]
features["Debt/Income"] = features["Total Debt"] / features["Income"].replace(0, 1)  # Avoid division by zero
features["Wealth Accumulation"] = features["Years Employed"] * features["Income"]
feature_names = features.columns.tolist()

# Preprocessing pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
preprocessor = ColumnTransformer(
    [("num", numeric_pipeline, feature_names)],
    verbose_feature_names_out=False
)
X_scaled = preprocessor.fit_transform(features)
scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]

# 2. KMeans with fixed k=4 for business needs
best_k = 4
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(X_scaled)
df["Segment"] = cluster_labels

# Calculate silhouette score
sil_score = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score (k={best_k}): {sil_score:.3f}")

# 3. Segment analysis
centroids_scaled = kmeans.cluster_centers_
centroids_unscaled = scaler.inverse_transform(centroids_scaled)
centroid_df = pd.DataFrame(centroids_unscaled, columns=feature_names)

# Business-friendly segment names
segment_names = {
    0: "Budget-Conscious Youth",
    1: "Affluent Professionals",
    2: "Debt-Prone Defaulters",
    3: "Established Savers"
}
df["Segment Name"] = df["Segment"].map(segment_names)

# 4. k-NN classification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, cluster_labels, test_size=0.3, random_state=42, stratify=cluster_labels
)

# Grid search for optimal k
param_grid = {"n_neighbors": list(range(1, 16, 2))}
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
best_knn = grid.best_estimator_
y_pred = best_knn.predict(X_test)
knn_acc = accuracy_score(y_test, y_pred)
print(f"k-NN Accuracy: {knn_acc:.3f} (best k={grid.best_params_['n_neighbors']})")

# 5. Visualizations - FIXED STYLE ISSUE
sns.set_style("whitegrid")  # Use Seaborn's whitegrid style
color_palette = sns.color_palette("tab10", best_k)

# A. Segment distribution
plt.figure(figsize=(10, 6))
segment_counts = df["Segment Name"].value_counts()
ax = sns.barplot(x=segment_counts.values, y=segment_counts.index, palette=color_palette)
plt.title("Customer Segment Distribution", fontsize=16, pad=15)
plt.xlabel("Number of Customers", fontsize=12)
plt.ylabel("")
for i, v in enumerate(segment_counts.values):
    ax.text(v + 3, i, f"{v} ({v / len(df):.1%})", va="center", fontsize=11)
plt.tight_layout()
plt.savefig(f"{fig_dir}/segment_distribution.png", dpi=300)
plt.close()

# B. Feature comparison (z-scores)
global_means = features.mean()
global_stds = features.std(ddof=0).replace(0, 1)
centroid_z = (centroid_df - global_means) / global_stds
centroid_z["Segment"] = [segment_names[i] for i in centroid_z.index]

plt.figure(figsize=(14, 8))
parallel_coordinates(
    centroid_z,
    "Segment",
    color=color_palette,
    alpha=0.8,
    linewidth=2.5
)
plt.title("Segment Profiles (Z-score Normalized)", fontsize=16, pad=15)
plt.ylabel("Standard Deviations from Mean", fontsize=12)
plt.xlabel("Features", fontsize=12)
plt.xticks(rotation=20)
plt.grid(alpha=0.3)
plt.legend(title="Segment", fontsize=10, loc="upper left")
plt.tight_layout()
plt.savefig(f"{fig_dir}/feature_profiles.png", dpi=300)
plt.close()

# C. PCA visualization
pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_.sum()

plt.figure(figsize=(10, 8))
for i, seg in enumerate(sorted(df["Segment"].unique())):
    mask = df["Segment"] == seg
    plt.scatter(
        proj[mask, 0],
        proj[mask, 1],
        label=segment_names[seg],
        alpha=0.7,
        s=50,
        edgecolor="w",
        linewidth=0.5,
        color=color_palette[i]
    )

# Add centroids
centers_proj = pca.transform(kmeans.cluster_centers_)
for i, (x, y) in enumerate(centers_proj):
    plt.scatter(x, y, s=300, marker="*", c="gold", edgecolor="k", zorder=10)
    plt.text(
        x,
        y + 0.15,
        segment_names[i],
        fontsize=11,
        weight="bold",
        ha="center",
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3")
    )

plt.title(f"Customer Segments (PCA: {explained_var:.1%} Variance Explained)", fontsize=16)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{fig_dir}/pca_segmentation.png", dpi=300)
plt.close()

# D. Confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred, normalize="true")
labels = [segment_names[i] for i in sorted(segment_names)]

sns.heatmap(
    cm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={"fontsize": 10}
)
plt.xticks(rotation=30, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.title(f"k-NN vs KMeans Agreement (Accuracy: {knn_acc:.2f})", fontsize=14)
plt.xlabel("Predicted Segment", fontsize=12)
plt.ylabel("True Segment", fontsize=12)
plt.tight_layout()
plt.savefig(f"{fig_dir}/knn_confusion_matrix.png", dpi=300)
plt.close()

# 6. Segment summary
segment_summary = []
for seg in sorted(centroid_df.index):
    size = (df["Segment"] == seg).sum()
    segment_summary.append({
        "Segment ID": seg,
        "Name": segment_names[seg],
        "Size": size,
        "Percentage": f"{size / len(df):.1%}",
        **centroid_df.loc[seg].to_dict()
    })

summary_df = pd.DataFrame(segment_summary)
summary_df.to_csv(summary_path, index=False)
print(f"Segment summary saved to {summary_path}")


# 7. New customer assignment
def explain_new_customer(customer_features):
    """Assign segment to new customer with detailed explanation"""
    new_df = pd.DataFrame([customer_features])
    scaled_new = preprocessor.transform(new_df)
    segment_num = best_knn.predict(scaled_new)[0]
    seg_name = segment_names[segment_num]

    # Distance to centroids
    dists = np.linalg.norm(scaled_new - kmeans.cluster_centers_, axis=1).flatten()
    dist_df = pd.DataFrame({
        "Segment": [segment_names[i] for i in range(best_k)],
        "Distance": dists
    }).sort_values("Distance")

    # Feature comparison
    centroid = centroid_df.loc[segment_num]
    diffs = (new_df.iloc[0] - centroid) / global_stds.replace(0, 1)
    notable = diffs.abs().nlargest(3)
    explanations = [
        f"{feat}: {new_df.iloc[0][feat]:.1f} vs typical {centroid[feat]:.1f} (z={diffs[feat]:.2f})"
        for feat in notable.index
    ]

    return {
        "assigned_segment": seg_name,
        "segment_distances": dist_df,
        "key_differences": explanations
    }


# Example usage
new_customer = {
    "Age": 28,
    "Edu": 14,
    "Years Employed": 3,
    "Income": 45000,
    "Card Debt": 3500,
    "Other Debt": 8000,
    "Defaulted": 0,
    "DebtIncomeRatio": 0.25,
    "Total Debt": 11500,
    "Debt/Income": 0.26,
    "Wealth Accumulation": 135000
}

result = explain_new_customer(new_customer)
print("\nNew Customer Assignment:")
print(f"Segment: {result['assigned_segment']}")
print("Distances to segments:")
print(result["segment_distances"])
print("\nKey differences:")
print("\n".join(result["key_differences"]))

print("\nAnalysis Complete!")
print(f"Visualizations saved to: {fig_dir}/")
print(f"Segment summary saved to: {summary_path}")