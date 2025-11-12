import pickle
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

done_count = 200
todo_count = 395 - done_count
compress_rate = 0.4
to_analyze = True

with open('lb.pickle', 'rb') as handle:
    lb = pickle.load(handle)

def calculate_difference(file1, file2):
    with open(file1, 'rb') as file:
        list1 = pickle.load(file)
    with open(file2, 'rb') as file:
        list2 = pickle.load(file)
    
    differences = []
    # print(len(list1), len(list2))
    for i in range(todo_count):
        difference = abs(list1[i] - list2[i])
        differences.append(difference)
    
    return differences

def run_all():
    print(f'testing compress_rate = {compress_rate}')
    current_file = os.path.basename(__file__)

    exclude = ["test.py", "autoencoder.py", "analysis.py"]
    for file in os.listdir('.'):
        if file.endswith('.py') and file != current_file and (file not in exclude):
            print(f'Running {file}...')
            subprocess.run(['python', file, str(compress_rate), str(to_analyze)])
    
    base_file = 'full.pickle'
    lb_file = 'lb.pickle'
    differences_dict = {}
    for file in os.listdir('.'):
        if file.endswith('.pickle') and file != base_file and file != lb_file and file != 'embeddings.pickle':
            differences = calculate_difference(base_file, file)
            differences_dict[file] = differences

    return differences_dict

if __name__ == '__main__':

    import sys
    
    if len(sys.argv) == 2 and sys.argv[1] == 'analysis':
        to_analyze = True
        run_all()
    elif len(sys.argv) == 2 and sys.argv[1] == 'compare':
        to_analyze = False
        differences_dict = run_all()

        base_file = 'full.pickle'
        colors = ['red', 'green', 'orange']
        labels = ['sample.pickle', 'pca.pickle', 'anneal.pickle']

        for i in range(len(labels)):
            with open(base_file, 'rb') as hd:
                list1 = pickle.load(hd)
            with open(labels[i], 'rb') as hd:
                list2 = pickle.load(hd)
            # print(list1)
            # print(list2)
            plt.scatter(list1, list2, c=colors[i], label=labels[i][:-7], s=5)
        
        x = np.linspace(0.2, 0.9, 100)
        y = x
        plt.plot(x, y, label='y = x', linestyle='-', linewidth=0.5)

        plt.title(f"Estimated Accuracy vs True Accuracy")
        plt.xlabel("True Acc")
        plt.ylabel("Estimated Acc")
        plt.legend()
        plt.savefig('Acc.pdf')
    else:
        to_analyze = False
        # 枚举 compress rate，分别 runall
        step = 0.07
        compress_rates = np.arange(0.05, 0.5, step)
        differences_dicts = {}
        for compress_rate in compress_rates:
            differences_dicts[compress_rate] = run_all()

        # 将结果可视化，横轴为压缩率，纵轴为误差区间(errorbar)，不同颜色代表不同文件
        colors = ['red', 'green', 'purple', 'blue', 'orange', 'yellow']
        labels = ['sample.pickle', 'pca.pickle', 'enc.pickle', 'bestsample.pickle', 'anneal.pickle', 'hiking.pickle']
        
        for i in range(len(labels)):
            x = compress_rates
            y = []
            for cr in compress_rates:
                differences = differences_dicts[cr][labels[i]]
                y.append(sum(x**2 for x in differences))
            plt.plot(x, y, label=labels[i][:-7], color=colors[i], marker='o')

            with open("mmlu-ext-output.txt", "a") as f:
                f.write(str(labels[i]) + '\n')
                f.write(str([round(t, 4) for t in y]) + '\n')

        plt.title("MMLU-ext")
        plt.xlabel("Compress Rate")
        plt.ylabel("Performance Est. Error")
        # plt.ylim(0, 0.1)
        plt.legend()
        plt.show()

        plt.savefig(f'mmlu-ext.pdf')