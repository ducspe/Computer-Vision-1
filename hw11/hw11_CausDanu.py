import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

ar1 = np.loadtxt("Aggregation.txt")
ar2 = ar1[:,0:2]
print(ar2)

fourmeans = KMeans(n_clusters=4, random_state=2019).fit(ar2)
fig1 = plt.figure("4 cluster figure", figsize=(8, 6))
plt.scatter(ar2[:, 0], ar2[:, 1], c=fourmeans.labels_.astype(float))

eightmeans = KMeans(n_clusters=8, random_state=2019).fit(ar2)
fig2 = plt.figure("8 cluster figure", figsize=(8, 6))
plt.scatter(ar2[:, 0], ar2[:, 1], c=eightmeans.labels_.astype(float))

twelvemeans = KMeans(n_clusters=12, random_state=2019).fit(ar2)
fig3 = plt.figure("12 cluster figure", figsize=(8, 6))
plt.scatter(ar2[:, 0], ar2[:, 1], c=twelvemeans.labels_.astype(float))

seveneightyeightmeans = KMeans(n_clusters=788, random_state=2019).fit(ar2)
fig4 = plt.figure("788 cluster figure", figsize=(8, 6))
plt.scatter(ar2[:, 0], ar2[:, 1], c=seveneightyeightmeans.labels_.astype(float))

plt.show()
'''
euclidean = []
for index, center in enumerate(seveneightyeightmeans.cluster_centers_):
    euclidean.append(center[0]**2 + center[1]**2)

print(len(list(set(euclidean))))
'''

kmeans = KMeans(n_clusters=788, random_state=7855).fit(ar2)

print(len(set(kmeans.labels_)))

'''
1) If we use a random seed of 2019, k=4 case seems to give the best result out of the options: [k=4, k=8, k=12] and k=12: the worst result out of the same options.

2) The labels seem to be different (i.e 788 unique values) for different random seeds (if I print the length of unique values of the "labels_" array)
But intuitively I would expect to have some overlap of the cluster centers in some unfortunate random initialization cases, which would then result in a smaller number of unique labels.
'''