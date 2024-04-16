import numpy as np
from PIL import Image
from torchvision.transforms import v2
from torchvision import transforms
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import CustomClass
from func_file import train_model, test_model, collate_fn

transformer = v2.Compose(transforms=
                                 [transforms.ToTensor(),
                                  v2.Resize(size=[64,128])
                                  ])

eng_only_vocab = CustomClass.Vocab()
eng_only_vocab.resetCode()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

def predict_test(model_path, check_id=0):
    model_name = model_path

    model = torch.load(model_name, map_location=device)

    img_path = './data/archive (3)/new_test/new_test'
    testFiles = os.listdir(img_path)

    img = Image.open(img_path + '/' + testFiles[check_id])
    label = testFiles[check_id].split('.jpg')[0]
    imgTS = transformer(img).unsqueeze(dim=0).to(device)
    img.close()

    pre = model(imgTS)

    print(torch.argmax(pre, dim=-1)[0])

    for id in torch.argmax(pre, dim=-1)[0]:
        letter = eng_only_vocab.decoder[id.item()]
        if letter != '<PAD>':
            print(letter, end=', ')

    print(f'\n{label}')

def predict_input(model_path, img):
    model_name = model_path

    model = torch.load(model_name)

predict_test('./data/backup_model/bbbest_model200.pkl', 112)
