import json
from random import shuffle
import math

import numpy as np
import torch
from transformers import BertTokenizer

YELP_DATA = "./data/yelp_review_training_dataset.jsonl"
def load_data():
    with open(YELP_DATA, 'r') as json_file:
        json_list = list(json_file)

    i = 0
    text, labels = [], []
    for json_str in json_list:
        if i == 500:
            break
        result = json.loads(json_str)
        text.append(result['text'])
        labels.append(result['stars'])

    return text, labels

def partition(text, labels, train_ratio=0.8):
    shuffled = list(zip(text, labels))
    shuffle(shuffled)
    validation_size = math.floor(len(text) * train_ratio)

    shuffled_data = [d for d, _ in shuffled]
    shuffled_labels = [l for _, l in shuffled]

    training_data = shuffled_data[:validation_size]
    training_labels = shuffled_labels[:validation_size]
    validation_data = shuffled_data[validation_size:]
    validation_labels = shuffled_labels[validation_size:]
    
    return training_data, training_labels, validation_data, validation_labels

def tokenize(text_batch):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
    return encoding

if __name__ == "__main__":
    text, labels = load_data()
    partition(text, labels)
    X_train, y_train, X_val, y_val = partition(text, labels)
    encoding = tokenize(X_train[batch_size * i: batch_size * (i + 1)])