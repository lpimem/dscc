from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
from skfuzzy.cluster import cmeans as fuzzy_kmeans
import numpy as np
from collections import Counter
from .kmeans import FuzzyKMeans


def label_by_cluster(X, cluster_method, n_clusters):
    """
        cluster given data using the 'cluster_method'
    """
    if cluster_method == "FuzzyKMeans":
        kvargs = {
            "k": n_clusters,
            "m": 2,
            "tol": 1e-4,
            "max_iter": 1000,
        }
        attr = FuzzyKMeans
    else:
        if cluster_method == "dbscan":
            kvargs = {
                "eps": .3
            }
        elif cluster_method == "SpectralClustering":
            kvargs = {
                "n_clusters": n_clusters,
                "eigen_solver": 'arpack',
                "affinity": 'nearest_neighbors'
            }
        elif cluster_method == "AgglomerativeClustering":
            connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
            connectivity = 0.5 * (connectivity + connectivity.T)
            kvargs = {
                "linkage": "average",
                "affinity": "cityblock",
                "n_clusters": n_clusters,
                "connectivity": connectivity
            }
        else:
            kvargs = {
                "n_clusters": n_clusters
            }
        attr = getattr(cluster, cluster_method)
    m = attr(**kvargs)
    try:
        print(X.shape)
        m.fit(X)
    except TypeError:
        m.fit(X.toarray())
    if hasattr(m, 'labels_'):
        return m.labels_.astype(np.int)
    else:
        return m.predict(X)


def map_labels(Y, clusters, Y_test):
    """
    Find relations between labels and Y by identifying the majority of the real labels of a cluster, and 
    convert Y_test to corresponding labels of the clusters in the same way
    """
    # labels of each collection
    collections = {}
    # map from cluster to label
    mapping = {}
    # all distinct clusters
    cluster_set = set([])
    for i in range(len(clusters)):
        if clusters[i] not in collections:
            collections[clusters[i]] = []
        collections[clusters[i]].append(Y[i])
        cluster_set.add(clusters[i])
    for c in cluster_set:
        mapping[c] = Counter(collections[c]).most_common()[0][0]
    # mapping from label to clusters
    reverse_map = {}
    for k, v in mapping.items():
        if v in reverse_map:
            print("[Warning] two clusters found for a single label")
            print(">>>>", mapping[v], "and", k, "for", v)
        reverse_map[v] = k
    for i in Y_test:
        if i not in reverse_map:
            print("[ERROR]label ", i, "is not mapped to any cluster")
    # return cluster numbers for each label in test set
    return [reverse_map[i] if i in reverse_map else clusters[0]
            for i in Y_test]
