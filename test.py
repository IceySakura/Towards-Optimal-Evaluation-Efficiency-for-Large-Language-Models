from main import done_count, todo_count, lb

def test_all(ques_ids, ques_sizes):
    scores = [0] * todo_count

    ques_count = 0
    for scenario in lb.keys():
        correctness = lb[scenario]['correctness']
        ques_id = ques_ids[scenario]
        ques_size = ques_sizes[scenario]
        
        for i in range(len(ques_id)):
            id = ques_id[i]
            size = ques_size[i]
            ques_count += size
            for k in range(todo_count):
                scores[k] += correctness[id][done_count + k] * size
        
        # print(f'{scenario} done')
        # print(scenario_scores)
    
    return [x / ques_count for x in scores]