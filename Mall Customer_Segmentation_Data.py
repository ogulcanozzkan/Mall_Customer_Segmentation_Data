import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

customer_data = pd.read_csv("Mall_Customers.csv")

customer_data.head()
customer_data.shape
customer_data.info()
customer_data.isnull().sum()

X = customer_data.iloc[:,[3,4]].values
print(X)

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++",random_state=42)
    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

sns.set()
plt.plot(range(1,11),wcss)
plt.title("The elbow point graph")
plt.xlabel("Number of Cluster")
plt.ylabel("WCSS")
plt.show(block = True)
plt.pause(5)

kmeans = KMeans(n_clusters=5,init="k-means++",random_state=0)
Y = kmeans.fit_predict(X)
print(Y)


plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='blue', label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show(block=True)
plt.pause(5)

