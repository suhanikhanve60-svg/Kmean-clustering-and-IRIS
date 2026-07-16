import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data=pd.read_csv(r"C:\Users\HP\Downloads\Book 3(Sheet1).csv")
print(data)
X=data[['X','Y']]
kmeans=KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
data['Cluster']=kmeans.labels_
print("\nClustered Data:")
print(data)
print("\nCentroids:")
print(kmeans.cluster_centers_)
plt.figure(figsize=(7,6))
plt.scatter(data['X'],data['Y'],c=data['Cluster'],cmap='viridis',s=120)
centroid = kmeans.cluster_centers_
plt.scatter(
    centroid[:,0],
      centroid[:,1],
      marker='X',
      s=250,
      color='red',
      label='Centroids'
)
for i in range(len(data)):
    plt.text(data['X'][i]+0.001,data['Y'][i]+0.001,str(data['ID'][i]))
plt.title('K-Means Clustering (K=2)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()