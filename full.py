import pickle
from main import lb
from test import test_all

if __name__ == '__main__':

    ques_ids = {}
    ques_sizes = {}
    for scenario in lb.keys():
        ques_count = lb[scenario]['correctness'].shape[0]
        ques_ids[scenario] = list(range(ques_count))
        ques_sizes[scenario] = [1] * ques_count

    scores = test_all(ques_ids, ques_sizes)
    # print(scores)

    with open('full.pickle', 'wb') as handle:
        pickle.dump(scores, handle)