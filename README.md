# Towards Optimal Evaluation Efficiency for Large Language Models (EMNLP2025)

This is the codebase for [Towards Optimal Evaluation Efficiency for Large Language Models (EMNLP2025)](https://aclanthology.org/2025.emnlp-main.716/)

build env:

```
conda create -n effi_eval python=3.10
conda activate effi_eval
pip install -r requirement.txt
```

run compare between methods:

```
cd ./src
python main.py
```

---

How to build my own leader board?

Here the presented `./src/lb.pickle` is MMLU result with 395 models, forming a 2d score matrix.

More leader boards are listed in ./pickle. You can check their format, and build your own leader board pickle for further analysis.