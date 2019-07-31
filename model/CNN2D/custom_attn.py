import torch
torch.manual_seed(123)
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class Atten(nn.Module):
    def __init__(self, num_classes, latent_dim=512, lstm_layers=1, hidden_dim=512, bidirectional=True):
        super(Atten, self).__init__()

        self._extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8)
        )

        self._rnnModule = nn.Sequential(
                nn.GRU(512, 512,num_layers=1 ,batch_first=False,bidirectional=True),
                # nn.GRU(512, 512, batch_first=False,bidirectional=True),
                #nn.LSTM(512, 512, batch_first=False, bidirectional=True),
                )

        self._classifier = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                         nn.ReLU(),
                                         nn.Dropout(),
                                         nn.Linear(in_features=512, out_features=256),
                                         nn.ReLU(),
                                         nn.Dropout(),
                                         nn.Linear(in_features=256, out_features=num_classes))

        self.attention_layer = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, 1)
        self.apply(self._init_weights)

    def forward(self, x):
        # x = torch.unsqueeze(x,1)
        x = self._extractor(x)
        x = x.permute(3,0,1,2)
        x = x.view(x.size(0), x.size(1), -1)
        x, hn = self._rnnModule(x)
        attention_w = F.softmax(self.attention_layer(x).squeeze(-1), dim=-1)
        # print(attention_w.shape)
        # print(attention_w.unsqueeze(-1).shape)
        # print(x.shape)
        x = torch.sum(attention_w.unsqueeze(-1) * x, dim=0)
        # print(x.shape)
        # x = x.permute(1, 2, 0)
        # x = x.view(x.size(0), -1)
        score = self._classifier(x)
        return score

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
