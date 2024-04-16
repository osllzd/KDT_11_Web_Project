### 모듈 로딩
import cgi, cgitb
import sys, codecs

cgitb.enable()

# Web 인코딩 설정 (한글 깨짐 해결)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

### text_pipeline 구현
import pickle
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

### 토크나이저 생성 (영어 기반)
tokenizer = get_tokenizer("basic_english")

entire_vocab: torchtext.vocab.Vocab

# 저장된 단어사전 불러오기
with open('model/vocab.pkl', 'rb') as f:
    entire_vocab = pickle.load(f)

# 텍스트 > 정수 인코딩
text_pipeline = lambda x: entire_vocab(tokenizer(x))

# 모델 정의
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LangModel(nn.Module):
    def __init__(self, VOCAB_SIZE, EMBEDD_DIM, HIDDEN_SIZE, NUM_CLASS):
        super().__init__()
        # 모델 구성 층 정의
        self.embedding = nn.EmbeddingBag(VOCAB_SIZE, EMBEDD_DIM, sparse=False)
        self.fc = nn.Linear(EMBEDD_DIM, NUM_CLASS)
        self.init_weights()
    
    # 가중치 초기화
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
    
    # 순방향 학습 진행
    def forward(self, text, offsets):
        x = self.embedding(text, offsets)
        x = self.fc(x)
        return x

# 판정
def predict(model, text, text_pipeline):
    # 평가 모드
    model.eval()
    answer_list = ['영어', '프랑스어', '독일어', '이탈리아어', '스페인어', '라틴어']

    with torch.no_grad():
        # 단어 인코딩 -> 텐서화
        text = torch.tensor(text_pipeline(text), dtype=torch.int64).to(device)
        offsets = torch.tensor([0]).to(device)
        pred = model(text, offsets)
        predicted_label = answer_list[pred.argmax(1).item()]
        # print(f'입력한 문장은 [ {predicted_label} ] 입니다.')
    return predicted_label

# 모델 로딩
MODEL = torch.load('model/latin_epoch10_model.pth')

# 웹 페이지의 form 태그 내의 input 태그 입력값을
# 가져와서 저장하고 있는 인스턴스 (딕셔너리 형식)
form = cgi.FieldStorage()     # 출력 : 웹 (터미널X)

# form['img_file'] => 딕셔너리
# value 를 취하면 값을 빼낼 수 있다.
if 'sentence' in form:
    msg = form['sentence'].value         # form.getvalue('message')
    answer = predict(MODEL, msg, text_pipeline)

## 요청에 대한 응답 HTML
print('Content-Type: text/html; charset=utf-8')    # HTML is following
print()
# print("<TITLE>CGI script ouput</TITLE>")
print(f"<h4>{msg}<h4>")
print(f"<H1>입력한 문장은 [ {answer} ] 입니다.</H1>")
# print(f"Hello, world! {form}")
