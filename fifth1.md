# Hierarchical Clustering
Same as K-Means but uses a **dendrogram** (tree diagram) to find K.
- Uses **linkage + dendrogram** instead of Elbow
- Uses **AgglomerativeClustering** instead of KMeans
- Everything else is the same as K-Means!
## Using CSV Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1: Load
df = pd.read_csv('Mall_Customers.csv')  # CHANGE filename
print(df.head())
df.dropna(inplace=True)

# Step 2: Pick features - CHANGE columns (same as K-Means)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
X_scaled = StandardScaler().fit_transform(X)
# Step 3: Dendrogram - find best K (cut where longest vertical gap is)
plt.figure(figsize=(12,5))
dendrogram(linkage(X_scaled, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()
# Step 4: Cluster + Plot - CHANGE k based on dendrogram
k = 3
model = AgglomerativeClustering(n_clusters=k)
df['Cluster'] = model.fit_predict(X_scaled)

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=df['Cluster'], cmap='viridis')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title('Hierarchical Clusters')
plt.colorbar()
plt.show()

print("Silhouette Score:", silhouette_score(X_scaled, df['Cluster']))
## Using Custom (Generated) Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1: Generate - CHANGE names and ranges
np.random.seed(42)
df = pd.DataFrame({
    'StudyHours': np.random.uniform(1, 10, 200),
    'Attendance': np.random.randint(40, 100, 200),
    'Marks': np.random.randint(20, 100, 200)
})
print(df.head())
df.hist(figsize=(10,6))
plt.tight_layout()
plt.show()

# Step 2: Dendrogram
X_scaled = StandardScaler().fit_transform(df)
plt.figure(figsize=(12,5))
dendrogram(linkage(X_scaled, method='ward'))
plt.title('Dendrogram')
plt.show()

# Step 3: Cluster + Plot
k = 3  # CHANGE from dendrogram
model = AgglomerativeClustering(n_clusters=k)
df['Cluster'] = model.fit_predict(X_scaled)

plt.scatter(df.iloc[:,0], df.iloc[:,1], c=df['Cluster'], cmap='viridis')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title('Hierarchical Clusters')
plt.colorbar()
plt.show()

print("Silhouette Score:", silhouette_score(X_scaled, df['Cluster']))