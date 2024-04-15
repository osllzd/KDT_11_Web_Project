import cgi, os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms
from CustomClass import Vocab, CustomModel

### ==> Client 요청 데이터 즉, Form 데이터 저장 인스턴스
form = cgi.FieldStorage()

### ==> 트랜스포머 정의
transformer = transforms.Compose(transforms=
                                 [transforms.ToTensor(),
                                  transforms.Resize(size=[64,128])
                                  ])

### ==> 모델 로딩
model_choice = 2
model_name_list = ['/bbbest_model100.pkl', '/bbbest_model200.pkl', '/bbbbest_model100.pkl']
pklfile = os.path.dirname(__file__) + model_name_list[model_choice]
model = torch.load(pklfile, map_location='cpu')
vocab = Vocab()
vocab.resetCode()

### ==> 이미지가 잘렸을 때 발생하는 에러 방지
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    ### ==> 이미지 추출 및 텐서화
    img_name = form['img'].filename
    img_file = form['img'].file
    save_path = f'./img/{img_name}'
    with open(file=save_path, mode='wb') as f:
        f.write(img_file.read())

    img = Image.open(save_path)
    imgTS = transformer(img).unsqueeze(dim=0)

    ### ==> 모델 예측
    pre = model(imgTS)
    pre_str = ''
    
    for id in torch.argmax(pre, dim=-1)[0]:
        letter = vocab.decoder[id.item()]
        if letter != '<PAD>':
            pre_str += letter

    result = pre_str

except Exception as e:
    img = None
    result = e

### ==> Web 브라우저 화면 출력 코드
# HTML 파일 읽기 -> body 문자열
if 'img' in form:
    filename = './result.html'
else:
    filename = './main.html'
with open(filename, 'r', encoding='utf-8') as f:
    # HTML Header
    print("Content-Type: text/html")
    print()

    # HTML Body
    print(f.read().format(result))
    print()
    if img:
        print(img.show())
