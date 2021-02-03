# FedNewsRec-EMNLP-Findings-2020
- Code of our paper "Privacy-Preserving News Recommendation Model Learning"

# Data Preparation
- If you want to test this project, you should download MIND-Small dataset in https://msnews.github.io/index.html
- Let data-root-path denote the root path of the embedding
- Files in the training dataset should be placed in root\_data\_path/train
- Files in the validation dataset should be placed in data-root-path/val
- We used the glove.840B.300d embedding vecrors in https://nlp.stanford.edu/projects/glove/
- The embedding file should be placed in embedding\_path\glove.840B.300d.txt

# Code Files
- preprocess.py: containing functions to preprocess the datasets
- utils.py: containg some util functions, such as evaluation matrics
- generator.py: containing data generator for model evaluation
- models.py: containing codes for implementing the base model of FedRec
- fl\_training.py: containing codes for federated model training
- Example.ipynb: containing codes for model training and evaluation

