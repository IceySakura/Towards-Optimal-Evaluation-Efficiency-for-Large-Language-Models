import pickle
import random
from main import lb, done_count, to_analyze
from test import test_all
from analysis import analyze
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 压缩率
import sys

# print(sys.argv)
if len(sys.argv) < 3:
    from main import compress_rate, to_analyze
else :
    compress_rate = float(sys.argv[1])
    to_analyze = sys.argv[2] == 'True'

# print(f'compress_rate = {compress_rate}, to_analyze = {to_analyze}')

# 获取聚类中心，找到 k 个 anchor point 对应的下标 
import numpy as np
from sklearn.cluster import KMeans

def get_centers(vecs, k, scenario):
    vecs = np.array(vecs)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(vecs)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    center_indices = []
    cluster_sizes = []
    for center in centers:
        distances = np.linalg.norm(vecs - center, axis=1)

        center_index = np.argmin(distances)
        cluster_size = np.sum(labels == kmeans.predict([center])[0])
        
        center_indices.append(center_index)
        cluster_sizes.append(cluster_size)

    # if k > 30:
    #     plt.figure(figsize=(8, 6))

    #     for cluster_idx in range(k):
    #         cluster_points = vecs[labels == cluster_idx]
    #         plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10)

    #     plt.scatter(centers[:, 0], centers[:, 1], c="red", marker=".", s=100, label="Centers")

    #     plt.title(f"KMeans Clustering with k={k}")
    #     plt.xlabel("X1")
    #     plt.ylabel("X2")
    #     plt.legend()
    #     plt.savefig(f'{scenario}.pdf')

    # print(f'k = {k}')
    # print(len(vecs))
    # print(center_indices, cluster_sizes)
    return center_indices, cluster_sizes

if __name__ == '__main__':

    ques_ids = {}
    ques_sizes = {}
    for scenario in lb.keys():

        ques_count = lb[scenario]['correctness'].shape[0]
        k = max(int(ques_count * compress_rate), 1)
        # print(ques_count, max(int(ques_count * compress_rate), 1))

        vecs = [row[:done_count] for row in lb[scenario]['correctness']]
        pca = PCA(n_components=2)
        vecs_pca = pca.fit_transform(vecs)
        # print(f"选择的主成分个数: {pca.n_components_}")
        # print(f"解释的方差比例: {pca.explained_variance_ratio_}")
        # print(f"累计方差比例: {np.cumsum(pca.explained_variance_ratio_)}")
        # print(vecs_pca)
        scenario_ids, cluster_sizes = get_centers(vecs_pca, k, scenario)

        ques_ids[scenario] = scenario_ids
        ques_sizes[scenario] = cluster_sizes
        # ques_sizes[scenario] = [ques_count / k] * 
    
    if to_analyze:
        # print(f'to_analyze = {to_analyze}')
        analyze('pca', ques_ids, ques_sizes)
    scores = test_all(ques_ids, ques_sizes)
    # print(scores)

    with open('pca.pickle', 'wb') as handle:
        pickle.dump(scores, handle)