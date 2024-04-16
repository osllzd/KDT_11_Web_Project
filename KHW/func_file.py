import numpy as np
import torch
import torch.nn as nn
from PIL import Image

def collate_fn(batch):
    features = []
    labels = []
    max_len = max([len(label) for _, label in batch])

    for f, t in batch:
        f = f.numpy()
        features.append(f)
        diff = max_len - len(t)
        if diff > 0:
            zero_pad = np.zeros((diff,), dtype='int32')
            labels.append(np.append(t, zero_pad))
        else:
            labels.append(t)

    x = torch.FloatTensor(features)
    y = torch.IntTensor(labels)

    return x, y


def train_model(model, optim, trainDL, validDL, device, epochs=100, schd=None):
    train_cost_list = []
    valid_cost_list = []
    train_acc_list = []
    valid_acc_list = []
    min_cost = 1000
    # torch.autograd.set_detect_anomaly(True)

    for e in range(1, epochs+1):
        print('Proceeding...', end='')
        model.train()
        for idx, (x, y) in enumerate(trainDL):
            x, y = x.to(device), y.to(device)
            train_h = model(x)
            # CTCloss에 넣을 때 shape는 무조건 (input lenght, batch size, number of character) 순이어야 하더라
            # 좆같은 파이토치 개새끼 진짜
            train_h = train_h.permute(1,0,2)
            # print('train', train_h.shape, y.shape)
    
            train_h_size = torch.IntTensor([train_h.size(0)] * trainDL.batch_size).to(device)
            target_len = torch.IntTensor([len(txt) for txt in y]).to(device)
            # print(train_h_size, train_h_size.shape, target_len)
            
            # z =  torch.randn_like(train_h)
            train_cost = nn.functional.ctc_loss(train_h, y, train_h_size, target_len, zero_infinity=True)
            # print(train_cost)

            optim.zero_grad()
            train_cost.backward()
            optim.step()

            if idx % 50 == 0:
                print('.', end='')

        model.eval()
        for idx, (x, y) in enumerate(validDL):
            x, y = x.to(device), y.to(device)
            valid_h = model(x)
            valid_h = valid_h.permute(1,0,2)
            # print('valid', valid_h.shape, y.shape)

            valid_h_size = torch.IntTensor([valid_h.size(0)] * validDL.batch_size).to(device)
            target_len = torch.IntTensor([len(txt) for txt in y]).to(device)

            valid_cost = nn.functional.ctc_loss(train_h, y, valid_h_size, target_len, zero_infinity=True)

            if idx % 50 == 0:
                print('.', end='')

        if schd:
            schd.step(valid_cost)

        print(f'\nEpoch [{e:5} / {epochs:5}] ------')
        print(f'train cost = {train_cost}, valid cost = {valid_cost}')
        train_cost_list.append(train_cost)
        valid_cost_list.append(valid_cost)

        if valid_cost < min_cost:
            min_cost = valid_cost
            torch.save(model, f'./best_model{e}.pkl')
        
    print('--- Model train completed ---')

    return train_cost_list, valid_cost_list, train_acc_list, valid_acc_list


def test_model(model, testDL, device):
    test_cost_list = []
    model.eval()
    print('Proceeding...', end='')
    for idx, (x, y) in enumerate(testDL):
        x, y = x.to(device), y.to(device)
        test_h = model(x)
        test_h = test_h.permute(1,0,2)

        test_h_size = torch.IntTensor([test_h.size(0)] * testDL.batch_size).to(device)
        target_len = torch.IntTensor([len(txt) for txt in y]).to(device)

        test_cost = nn.functional.ctc_loss(test_h, y, test_h_size, target_len, zero_infinity=True)

        if idx % 10 == 0:
            print('.', end='')

        print(f'Trial {idx:3} --- ')
        print(f'test cost = {test_cost}')
        test_cost_list.append(test_cost.item())

    return test_cost_list

def predict_model(model, img, device):
    model.eval()

    img.to(device)

    pre = model(img)

    return(pre)