import pickle
import random
from main import lb, done_count
from test import test_all

# 固定随机种子
random.seed(1452)

# 压缩率
import sys
if len(sys.argv) < 2:
    from main import compress_rate
else :
    compress_rate = float(sys.argv[1])

# 采样次数
sample_count = 100

if __name__ == '__main__':

    ques_ids = {}
    ques_sizes = {}
    for scenario in lb.keys():
        ques_count = lb[scenario]['correctness'].shape[0]
        k = max(int(ques_count * compress_rate), 1)

        # 计算一个不压缩时候的分数
        full_scores = [0.0] * done_count
        for id in range(ques_count):
            for j in range(done_count):
                full_scores[j] += lb[scenario]['correctness'][id][j]
        full_scores = [x / ques_count for x in full_scores]

        # 计算当前采样方式的 err
        def get_err(ids):
            sample_scores = [0.0] * done_count
            for i in range(k):
                id = ids[i]
                for j in range(done_count):
                    sample_scores[j] += lb[scenario]['correctness'][id][j]
            sample_scores = [x / k for x in sample_scores]

            err = 0.0
            for i in range(done_count):
                err += (full_scores[i] - sample_scores[i])**2
            return err
        
        best_ids = []
        best_sizes = [ques_count / k] * k
        best_err = 1e9
        for T in range(sample_count):
            # 多轮采样，和不压缩时候的分数做差
            ids = random.sample(range(ques_count), k)
            err = get_err(ids)

            if err < best_err:
                best_ids = ids
                best_err = err
        
        # print(f'{scenario} done. best_err = {best_err}')
        
        ques_ids[scenario] = best_ids
        ques_sizes[scenario] = best_sizes
        

    scores = test_all(ques_ids, ques_sizes)
    # print(scores)

    with open('bestsample.pickle', 'wb') as handle:
        pickle.dump(scores, handle)