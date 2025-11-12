import torch
import pickle
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import ot
from tqdm import tqdm

# MMLU 数据集
scenarios = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
embeddings = {}

def normalize(lst):
    total = sum(lst)
    return [x / total for x in lst]

def analyze(method, ques_ids, ques_sizes):
    # 用于计算 Wasserstein 距离
    x = []
    y = []
    p = []
    q = []

    # Get embeddings
    cnt = 30
    with open('embeddings.pickle', 'rb') as handle:
        embeddings = pickle.load(handle)
    for scenario in scenarios:
        # print(f'Getting {scenario}...')
        
        ss = f'harness_hendrycksTest_{scenario}_5'
        ques_id = ques_ids[ss]
        ques_size = ques_sizes[ss]
        # print(len(ques_id),len(ques_size))
        # print(len(x),len(p))

        ee = embeddings[scenario]
        pp = 0
        # print(len(ee))
        for i in range(len(ee)):
            embedding = ee[i]
            y.append(embedding)

            if i in ques_id:
                x.append(embedding)
                p.append(ques_size[pp])
                pp += 1
                # print(f'x.append({embedding})')
        
        q += [1.0] * len(ee)
        if len(p) < len(x) : 
            p.append(0.0)
        
        cnt -= 1
        if cnt == 0:
            break

    # sinkhorn 计算 Wasserstein 距离
    # x = np.array(x)
    # y = np.array(y)
    # p = normalize(p)
    # q = normalize(q)
    # print(f'x:{len(x)}, y:{len(y)}, p:{len(p)}, q:{len(q)}')

    # M = np.linalg.norm(x[:, None] - y[None, :], axis=2)
    # print(f'M fin')

    # distance = ot.sinkhorn2(p, q, M, reg=0.1)
    # with open(f'{method}-distance.txt', 'w') as f:
    #     f.write(f'{distance}')

    # pca
    # pca = PCA(n_components=2)
    # pca.fit(y)
    # x_pca = pca.transform(x)
    # y_pca = pca.transform(y)

    # 计算 Centroid Distance
    xmean = np.mean(np.array(x), axis=0)
    ymean = np.mean(np.array(y), axis=0)
    distance = round(np.linalg.norm(xmean - ymean), 4)
    with open(f'{method}-cent-distance.txt', 'w') as f:
        f.write(f'{distance}')

    # # 画图
    # plt.figure(figsize=(10, 10))
    # plt.scatter(y_pca[:, 0], y_pca[:, 1], alpha=0.5)
    # plt.scatter(x_pca[:, 0], x_pca[:, 1], alpha=0.7, c='red')
    # plt.title(f'{method}', fontsize=20)
    # plt.show()
    # plt.savefig(f'{method}-analysis.pdf')

if __name__ == '__main__':
    # 加载 sentence-transformers
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model.to(device)
    print(f'model loaded, using {device}')

    # 预处理 embedding
    for scenario in scenarios:
        print(f'Embedding {scenario}...')
        dataset = load_dataset("cais/mmlu", scenario)['test']
        embeddings[scenario] = []
        for ques in dataset:
            embedding = model.encode(ques['question'])
            embeddings[scenario].append(embedding)
    
    # 存为 pickle
    with open('embeddings.pickle', 'wb') as handle:
        pickle.dump(embeddings, handle)