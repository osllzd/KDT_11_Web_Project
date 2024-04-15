'''
    사용자 정의 클래스 전용 파일
'''
import numpy as np
import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Vocab():
    def __init__(self) -> None:
        self.encoder = {}
        self.decoder = {}
        
    def __len__(self):
        return len(self.encoder)

    def resetCode(self):
        self.encoder.clear()
        self.decoder.clear()
        self.encoder['<PAD>'] = 0
        self.encoder['<UNK>'] = 1
        self.decoder[0] = '<PAD>'
        self.decoder[1] = '<UNK>'

        for idx, letter in enumerate(string.punctuation+string.ascii_letters+string.digits):
            self.encoder[letter] = idx+2
            self.decoder[idx+2] = letter

    def appendCode(self, letters):
        last = max(self.encoder.values) if len(self) > 0 else -1
        
        for idx, letter in enumerate(letters):
            if letter not in self.encoder:
                self.encoder[letter] = last + idx + 1
                self.decoder[last + idx + 1] = letter


    def idToAl(self, id):
        return self.decoder[id]
    
    def AlToid(self, letter):
        return self.encoder[letter]


class CustomDataset(Dataset):
    punc_dict = {' ':'<PAD>', '©':'<UNK>', '°':'<UNK>', '—':'-', '‘':'\'', '’':'\'', '“':'\"', '”':'\"'}
    def __init__(self, x, y, vocab):
        self.feature = x
        self.target = y
        self.vocab = vocab

    def __len__(self):
        return self.feature.shape[0]

    def __getitem__(self, idx):
        label = self.target[idx]
        label = (list(map(lambda x:self.punc_dict[x] if x in self.punc_dict.keys() else x, list(label))))
        label = (list(map(lambda x:self.vocab.AlToid(x), list(label))))
        
        return self.feature[idx], np.array(label, dtype='int32')
    
    def toDataLoader(self, batch, sampler=None, drop=True):
        return DataLoader(self, batch_size=batch, shuffle=True,
                          sampler=sampler, drop_last=drop)
    

class CustomModel(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=64,
                      kernel_size=(3,3), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(3,3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3,3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=(3,3), padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=(3,3), padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.gru = nn.GRU(64, 64, 3, batch_first=True)

        self.fc1 = nn.Linear(16*256, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.CNN(x)
        x = x.view(x.shape[0], x.shape[-1], -1)
        x = self.fc1(x)
        x, _ = self.gru(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.log_softmax(x, dim=-1)
        return x
