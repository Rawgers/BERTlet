from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import torch as th
from torch import nn

class LanguageModel(nn.Module):
    def __init__(self, bert_model_link, output_shape, lstm_hidden_size=256,
            batch_first=True, num_lstm_layers=1, use_residue=False, dropout=0):
        super().__init__()
        self.use_residue = use_residue
        self.bert = BertModel.from_pretrained(bert_model_link)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, lstm_hidden_size,
                batch_first=batch_first, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        linear_input_shape = lstm_hidden_size + self.bert.config.hidden_size if \
                use_residue else lstm_hidden_size

        self.linear = nn.Linear(linear_input_shape, output_shape)

    def forward(self,x):
        bert_output = self.bert(**x)
        lstm_output,_ = self.lstm(bert_output.last_hidden_state)
        dropout_output = self.dropout(lstm_output)
        last_dropout = dropout_output[:, -1, :].squeeze()
        input_to_linear = th.hstack((last_dropout, bert_output.pooler_output)) if self.use_residue else last_dropout
        linear_output = self.linear(input_to_linear)
        return linear_output

