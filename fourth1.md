# K-Means Clustering
Groups data into K clusters. No target column (unsupervised).
- **Elbow Method** = find best K (plot inertia, pick the bend)
- **Silhouette Score** = closer to 1 = better
## Using CSV Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Step 1: Load
df = pd.read_csv('Mall_Customers.csv')  # CHANGE filename
print(df.head())
df.dropna(inplace=True)

# Step 2: Pick feature columns - CHANGE these
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
# Examples: Iris=['SepalLengthCm','PetalLengthCm'], CreditCard=['BALANCE','PURCHASES']

X_scaled = StandardScaler().fit_transform(X)
# Step 3: Elbow Method - find best K
inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_ for k in range(1,11)]
plt.plot(range(1,11), inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
# Step 4: Train with chosen K - CHANGE k based on elbow
k = 3
model = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = model.fit_predict(X_scaled)

# Step 5: Plot clusters
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=df['Cluster'], cmap='viridis')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title('Clusters')
plt.colorbar()
plt.show()

print("Silhouette Score:", silhouette_score(X_scaled, df['Cluster']))
## Using Custom (Generated) Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

# Step 2: Scale + Elbow
X_scaled = StandardScaler().fit_transform(df)
inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_ for k in range(1,11)]
plt.plot(range(1,11), inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Step 3: Cluster + Plot
k = 3  # CHANGE from elbow
model = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = model.fit_predict(X_scaled)

plt.scatter(df.iloc[:,0], df.iloc[:,1], c=df['Cluster'], cmap='viridis')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title('Clusters')
plt.colorbar()
plt.show()

print("Silhouette Score:", silhouette_score(X_scaled, df['Cluster']))