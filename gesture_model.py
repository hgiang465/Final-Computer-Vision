"""Mô hình chuỗi cho MODULE 2 (LSTM / GRU) bằng PyTorch."""
import torch
import torch.nn as nn


class SeqClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, arch="gru", hidden=128, layers=2, dropout=0.3):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[arch]
        self.rnn = rnn_cls(input_dim, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):                      # x: (B, T, D)
        out, _ = self.rnn(x)
        feat = torch.cat([out[:, -1], out.mean(dim=1)], dim=1)   # trạng thái cuối + trung bình theo thời gian
        return self.fc(self.drop(feat))
