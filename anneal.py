import pickle
import random
import math
from main import lb, done_count, to_analyze
from test import test_all
from analysis import analyze

# 固定随机种子
random.seed(1452)

# 压缩率
import sys
if len(sys.argv) < 2:
    from main import compress_rate, to_analyze
else :
    compress_rate = float(sys.argv[1])
    to_analyze = sys.argv[2] == 'True'

# 采样次数
sample_count = 100

# 在当前组合的邻域中生成一个新组合
def generate_neighbor(current, n):
    new_combination = current[:]
    # 随机替换一个元素
    replace_index = random.randint(0, len(current) - 1)
    new_element = random.choice([x for x in range(n) if x not in current])
    new_combination[replace_index] = new_element
    return new_combination

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

        ids = random.sample(range(ques_count), k)
        err = get_err(ids)

        Temp = 100
        alpha = 0.95
        for T in range(sample_count):
            # 多轮采样，和不压缩时候的分数做差
            
            new_ids = generate_neighbor(ids, ques_count)
            new_err = get_err(new_ids)

            delta_y = new_err - err

            # 判断是否接受邻域解
            if delta_y < 0 or random.random() < math.exp(-delta_y / Temp):
                ids = new_ids
                err = new_err
            # 更新最优解
            if err < best_err:
                best_ids = ids
                best_err = err
            
            Temp *= alpha

            # print(f'T = {Temp}, err = {err}, best_err = {best_err}')

        # print(f'{scenario} done.\nbest_err = {best_err}')
        
        ques_ids[scenario] = best_ids
        ques_sizes[scenario] = best_sizes

    if to_analyze:
        analyze('anneal', ques_ids, ques_sizes)
    scores = test_all(ques_ids, ques_sizes)
    # print(scores)

    with open('anneal.pickle', 'wb') as handle:
        pickle.dump(scores, handle)