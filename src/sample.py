import pickle
import random
from main import lb
from test import test_all

# 固定随机种子
random.seed(1452)

# 压缩率
import sys

if len(sys.argv) < 3:
    from main import compress_rate, to_analyze
else :
    compress_rate = float(sys.argv[1])
    to_analyze = sys.argv[2] == 'True'

if __name__ == '__main__':

    ques_ids = {}
    ques_sizes = {}
    for scenario in lb.keys():
        ques_count = lb[scenario]['correctness'].shape[0]
        k = max(int(ques_count * compress_rate), 1)
        # print(ques_count, max(int(ques_count * compress_rate), 1))
        
        ques_ids[scenario] = random.sample(range(ques_count), k)
        ques_sizes[scenario] = [ques_count / k] * k

    scores = test_all(ques_ids, ques_sizes)
    # print(scores)

    with open('sample.pickle', 'wb') as handle:
        pickle.dump(scores, handle)