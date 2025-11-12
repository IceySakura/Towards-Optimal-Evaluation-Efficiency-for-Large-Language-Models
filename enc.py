import numpy as np
import pickle
import torch
from main import lb, done_count
from test import test_all
from autoencoder import Autoencoder

# 压缩率
import sys
if len(sys.argv) < 2:
    from main import compress_rate
else :
    compress_rate = float(sys.argv[1])

# 获取聚类中心，找到 k 个 anchor point 对应的下标 
import numpy as np
from sklearn.cluster import KMeans
def get_centers(vecs, k):
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

    # print(f'k = {k}')
    # print(len(vecs))
    # print(center_indices, cluster_sizes)
    return center_indices, cluster_sizes

if __name__ == '__main__':
    model = Autoencoder(200, 20)
    model.load_state_dict(torch.load('model.pth', weights_only=True))
    model.eval()

    ques_ids = {}
    ques_sizes = {}
    for scenario in lb.keys():
        ques_count = lb[scenario]['correctness'].shape[0]
        k = max(int(ques_count * compress_rate), 1)
        # print(ques_count, max(int(ques_count * compress_rate), 1))

        vecs = [row[:done_count] for row in lb[scenario]['correctness']]
        vecs = np.array(vecs, dtype=np.float32)
        vecs_tensor = torch.tensor(vecs)
        vecs_enc = model.encoder(vecs_tensor).detach().numpy().tolist()

        scenario_ids, cluster_sizes = get_centers(vecs_enc, k)

        ques_ids[scenario] = scenario_ids
        ques_sizes[scenario] = cluster_sizes
        # ques_sizes[scenario] = [ques_count / k] * k

    scores = test_all(ques_ids, ques_sizes)
    # print(scores)

    with open('enc.pickle', 'wb') as handle:
        pickle.dump(scores, handle)