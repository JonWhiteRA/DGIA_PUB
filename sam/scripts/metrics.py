from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import numpy as np

METRICS_THRESHOLDS = {
    'Silhouette'        : [0.03, 0.07, 0.23, 0.67],
    'Calinski-Harabasz' : [2.24, 5.59, 12.67, 54.84],
    'Dunn Index'        : [0.93, 1.01, 1.23, 2.95],
    'Davies-Bouldin'    : [0.23, 0.81, 1.77, 3.47],
    'Cohesion'          : [0, 52.66, 19.76, 25.55],
    'Separation'        : [30.63, 47.84, 73.6, 126.25],
    'Xie-Beni Index'    : [0.2, 0.42, 0.85, 1.00]
}

# Generated
def dunn_index(data, labels):
    distances = pairwise_distances(data)
    intra_cluster_distances = []
    inter_cluster_distances = []

    unique_labels = np.unique(labels)

    # Calculate intra-cluster distances
    for label in unique_labels:
        cluster_points = data[labels == label]
        if len(cluster_points) > 1:
            intra_cluster_distances.append(np.mean(pairwise_distances(cluster_points)))

    # Calculate inter-cluster distances
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            cluster_i = data[labels == unique_labels[i]]
            cluster_j = data[labels == unique_labels[j]]
            inter_cluster_distances.append(np.mean(pairwise_distances(cluster_i, cluster_j)))

    # Calculate Dunn Index
    if inter_cluster_distances and intra_cluster_distances:
        return min(inter_cluster_distances) / max(intra_cluster_distances)
    else:
        return 0
    
# Generated 
def xie_beni_index(X, clusters):
    n_clusters = len(set(clusters))
    distances = pairwise_distances(X)
    
    # Cohesion: Within-cluster variance
    cohesion = 0
    for i in range(n_clusters):
        cluster_points = X[clusters == i]
        if len(cluster_points) > 1:
            cohesion += np.mean(pairwise_distances(cluster_points))  # Average distance in cluster
    cohesion /= n_clusters

    # Separation: Minimum distance between clusters
    separation = float('inf')
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cluster_i_points = X[clusters == i]
            cluster_j_points = X[clusters == j]
            try:
                separation_distance = np.mean(pairwise_distances(cluster_i_points, cluster_j_points))
            except ValueError:
                return None
            separation = min(separation, separation_distance)

    return cohesion / separation if separation > 0 else np.inf

# Generated
def cohesion(data, labels):
    n_clusters = len(set(labels))
    cohesion = []
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        if len(cluster_points) > 1:
            cohesion.append(np.mean(pairwise_distances(cluster_points)))
        else:
            cohesion.append(0)

    return np.mean(cohesion)

# Generated
def separation(data, labels):
    n_clusters = len(set(labels))
    separation = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cluster_i_points = data[labels == i]
            cluster_j_points = data[labels == j]
            try:
                separation.append(np.mean(pairwise_distances(cluster_i_points, cluster_j_points)))
            except ValueError:
                return None

    return np.mean(separation)

# Generated
def calculate_inertia(data, labels):
    inertia = 0
    for label in np.unique(labels):
        cluster_points = data[labels == label]
        if len(cluster_points) > 1:
            center = np.mean(cluster_points, axis=0)
            inertia += np.sum(pairwise_distances(cluster_points, [center])**2)
    return inertia

# Generated - always returns a huge negative value :(
def gap_statistic(data, labels, n_references=10):
    # Store the original inertia for each k
    original_inertia = []
    k_max = len(set(labels))
    
    # Fit Agglomerative Clustering for original data for k = 1 to k_max
    for k in range(1, k_max + 1):
        inertia = calculate_inertia(data, labels)
        original_inertia.append(inertia)

    # Store the gaps for each k
    gaps = []

    # Generate random reference data and calculate the inertia for each reference
    for k in range(1, k_max + 1):
        reference_inertia = []
        
        for _ in range(n_references):
            # Generate random data uniformly distributed
            random_data = np.random.rand(*data.shape)
            clustering = AgglomerativeClustering(n_clusters=k)
            labels = clustering.fit_predict(random_data)
            inertia = calculate_inertia(random_data, labels)
            reference_inertia.append(inertia)

        # Calculate the average inertia for reference data
        mean_reference_inertia = np.mean(reference_inertia)

        # Calculate the Gap statistic
        gap = mean_reference_inertia - original_inertia[k - 1]
        gaps.append(gap)

    return max(gaps)

# Create a score for the metric
def score(x, scores):
    if x is None:
        return 0
    elif x <= scores[0]:
        points = 0
    elif x <= scores[1]:
        points = 25
    elif x <= scores[2]:
        points = 50
    elif x <= scores[3]:
        points = 75
    else:
        points = 100
    return points

# Grade cluster algorithm
def grade(s, db, ch, d, coh, sep, x):
    # 100 points possible
    points = 0
    errored = 0
    # silhouette score: (-1, 1), want closer to 1
    points += score(s, METRICS_THRESHOLDS['Silhouette'])
    # # davies-bouldin: want nearly 0 score
    if db is not None:
        points += 100 - score(db, METRICS_THRESHOLDS['Davies-Bouldin'])
    # calinski-harabasz: higher is better!
    points += score(ch, METRICS_THRESHOLDS['Calinski-Harabasz'])
    # dunn index: higher is better, want > 1
    points += score(d, METRICS_THRESHOLDS['Dunn Index'])
    # cohesion: lower is better
    points += 100 - score(coh, METRICS_THRESHOLDS['Cohesion'])
    # separation: higher is better
    separation = score(sep, METRICS_THRESHOLDS['Separation'])
    points += separation
    # xie-beni: lower is better
    xie = score(x, METRICS_THRESHOLDS['Xie-Beni Index'])
    points += 100 - xie

    if xie is None:
        errored += 1
    if separation is None:
        errored += 1
    
    points = points / (len(set(METRICS_THRESHOLDS.keys())) - errored)

    if points < 50:
        return (points, 'F')
    elif points < 60:
        return (points, 'E')
    elif points < 70:
        return (points, 'D')
    elif points < 80:
        return (points, 'C')
    elif points < 90:
        return (points, 'B')
    else:
        return (points, 'A')

# Calculate metrics for algorithm
def calculate_metrics(data, labels):
    # Need to have at least 2 clusters
    if len(set(labels)) < 2:
        return (None, None, None, None, None, None, None)
    else:
        silhouette = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        calinski =  calinski_harabasz_score(data, labels)
        dunn = dunn_index(data, labels)
        coh = cohesion(data, labels)
        sep = separation(data, labels)
        x = xie_beni_index(data, labels)
        return (silhouette, davies_bouldin, calinski, dunn, coh, sep, x)