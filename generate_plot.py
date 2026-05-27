import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("data/penguins.csv")
# Preprocess data
data = data.dropna()
features = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)

data["cluster"] = kmeans.fit_predict(scaled_features)

# Map cluster labels to species names
cluster_to_species = {
    0: "Adelie",
    1: "Chinstrap",
    2: "Gentoo"
}
data["species"] = data["cluster"].map(cluster_to_species)

# Plot
sns.set_theme(style="whitegrid")

plt.figure(figsize=(9, 6))

ax = sns.scatterplot(
    data=data,
    x="bill_length_mm",
    y="flipper_length_mm",
    hue="species",
    style="species",
    s=90,
)

ax.set_title("Penguin Species Clusters", fontsize=16, weight="bold")
ax.set_xlabel("Bill Length (mm)")
ax.set_ylabel("Flipper Length (mm)")

plt.legend(title="Species")
plt.tight_layout()

plt.savefig("static/images/species_clusters.png", dpi=200)
plt.close()