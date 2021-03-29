import json
from random import shuffle
import math

import torch as th
from transformers import BertTokenizer

YELP_DATA = "data/yelp_review_training_dataset.jsonl"

def load_data(root_dir, mode="classification"):
    path = root_dir + YELP_DATA
    with open(path, 'r') as json_file:
        json_list = list(json_file)

    text, labels = [], []
    for json_str in json_list:
        result = json.loads(json_str)
        text.append(result['text'])
        labels.append(result['stars'])

    if mode == "classification":
        labels = th.LongTensor(labels) - 1
    elif mode == "regression":
        labels = th.tensor(labels)

    return text, labels

def partition(text, labels, train_ratio=0.8):
    shuffled = list(zip(text, labels))
    shuffle(shuffled)
    train_size = math.floor(len(text) * train_ratio)

    shuffled_data = [d for d, _ in shuffled]
    shuffled_labels = [l for _, l in shuffled]

    training_data = shuffled_data[:train_size]
    training_labels = th.stack(shuffled_labels[:train_size])
    validation_data = shuffled_data[train_size:]
    validation_labels = th.stack(shuffled_labels[train_size:])
    
    return training_data, training_labels, validation_data, validation_labels

def tokenize(text_batch, model_link='bert-base-uncased', max_length=300):
    tokenizer = BertTokenizer.from_pretrained(model_link)
    encoding = tokenizer(text_batch, return_tensors='pt', padding='max_length', max_length=300, truncation=True)
    return encoding

if __name__ == "__main__":
    text, labels = load_data("./")
    X_train, y_train, X_val, y_val = partition(text, labels)
    encoding = tokenize(X_train[:10], max_length=300)