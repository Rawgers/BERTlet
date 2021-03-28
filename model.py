from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import torch as th
from torch import nn

class LanguageModel(nn.Module):
    def __init__(self, bert_model_link, output_shape):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_model_link)
        self.hidden_size = self.bert.config.hidden_size
        self.linear = nn.Linear(self.hidden_size, output_shape)

    def forward(self,x):
        bert_output = self.bert(**x)
        linear_output = self.linear(bert_output.pooler_output)
        return linear_output

