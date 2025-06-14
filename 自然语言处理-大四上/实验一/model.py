# -*- coding: UTF-8 -*-
"""
@Project ：SA 
@File ：model.py
@Author ：AnthonyZ
@Date ：2024/10/9 14:50
"""

import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_filters,
        filter_sizes,
        output_dim,
        dropout_rate,
        pad_index,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=fs)
                for fs in filter_sizes
            ]
        )
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids):
        embedded = self.dropout(self.embedding(ids))

        # embedded = [batch size, seq len, embedding dim] -> [batch size, embedding dim, seq len]
        embedded = embedded.permute(0, 2, 1)

        # 使用定义的卷积层对文本特征进行提取，卷积层之间需要加入激活函数
        conved = [torch.relu(conv(embedded)) for conv in self.convs]

        # 对提取的特征进行最大池化
        pooled = [torch.max(conv, dim=2)[0] for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=-1))

        # 利用线性层，将获得的特征信息进行分类
        prediction = self.fc(cat)

        return prediction


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, 
                            bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids):
        
        embedded = self.embedding(ids)
        
        embedded = self.dropout(embedded)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        output = self.fc(hidden)
        
        return output