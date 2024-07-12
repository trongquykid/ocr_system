from VietnameseOcrCorrection.tool.predictor import Predictor
from VietnameseOcrCorrection.tool.utils import extract_phrases
import time
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

model_predictor = Predictor(device='cpu', model_type='seq2seq', weight_path='VietnameseOcrCorrection/weights/seq2seq_0.pth')

text = ['3. Trong quá trình thực hiện Thông tư này, nếu có khó khăn, vướng mắc,', 'Công an các đơn vị, địa phương và cơ quan, tổ chức, cá nhân có liên quan báo', 'cáo về Bộ Công an (qua Cục Cảnh sát quản lý hành chính về trật tự xã hội) để có', 'hương dẫn kịp thời.Ng', 'BỘ TRƯỞNG',
         'Nơi nhận:', '- Các đồng chí Thứ trường;', '- Các đơn vị trực thuộc Bộ Công an;', 'UU', '- Cộng an tinh, thành phố trực thuộc trung ương;', '- Cổng Thông tin điện tử Chính phủ;', '- Cộng Thông tin điện từ Bộ Công an;', '- Công báo;', '- Lưu: VT, C06 (TTDLDC).', ' Đại tướng Tô Lâm ']

def check_correct_ocr(unacc_paragraphs):

    text_correct = []
    for i, p in enumerate(unacc_paragraphs):
        print(p)
        if count_words(p) > 1:
            outs = model_predictor.predict(p.strip(), NGRAM=6)
            text_correct.append(outs)
        else:
            text_correct.append(p)

    return text_correct

def count_words(sentence):
    words = nltk.word_tokenize(sentence)  # Tách câu thành các từ
    words = [word for word in words if word.isalnum()]  # Loại bỏ dấu chấm câu
    return len(words)

check_correct_ocr(text)